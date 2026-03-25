{
  description = "YOLO26x-v1 — reproducible dev + model-download environments";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # ── Python version — pinned to 3.12 ──────────────────────────────────
        # 3.13 is NOT used: numpy 1.26.x (required by ultralytics 8.3)
        # dropped Py 3.13 support. Bump BOTH here and in requirements.txt
        # if upgrading numpy to 2.x after verifying ultralytics compatibility.
        python = pkgs.python312;

        # ── Shared packages present in both shells ───────────────────────────
        commonPythonPkgs = ps: with ps; [
          huggingface-hub
          numpy
          pillow
          tqdm
        ];

        # ── Standalone huggingface-cli script ────────────────────────────────
        # Using a proper derivation instead of a shellHook alias so it works
        # in non-interactive shells, scripts, and `nix develop --command ...`.
        hf-cli = pkgs.writeShellScriptBin "hf-cli" ''
          exec ${python.withPackages commonPythonPkgs}/bin/python \
            -m huggingface_hub.commands.huggingface_cli "$@"
        '';

        # ── Model weights (Nix-native, hash-pinned) ──────────────────────────
        # Fetch directly into the Nix store so `nix build .#model` is
        # reproducible and CI never hits the network twice.
        # To update: nix-prefetch-url <url> --type sha256
        modelWeights = pkgs.fetchurl {
          name   = "weapon-yolo26x-best.pt";
          url    = "https://huggingface.co/HaiderKhan6410/weapon-yolo26x/resolve/main/best.pt";
          # Replace with real hash after first download:
          #   nix store prefetch-file <url>
          sha256 = pkgs.lib.fakeSha256;
        };

      in {

        # ── nix build .#model ─────────────────────────────────────────────────
        # Copies weights into result/model/ — useful in CI or Docker builds.
        packages.model = pkgs.runCommand "yolo26x-weights" {} ''
          mkdir -p $out/model
          cp ${modelWeights} $out/model/best.pt
        '';

        # ── nix develop .#download ────────────────────────────────────────────
        # Minimal shell: only what you need to pull weights from HF.
        # Keep this separate from the dev shell so CI can use it without
        # pulling in the full ML stack.
        devShells.download = pkgs.mkShell {
          name     = "yolo26x-download";
          packages = [
            (python.withPackages commonPythonPkgs)
            hf-cli
          ];

          shellHook = ''
            export HF_HOME="$PWD/.cache/huggingface"
            export PYTHONDONTWRITEBYTECODE=1

            # Validate env on entry — fail loud, not at runtime
            python - <<'EOF'
import sys, importlib.metadata as m
required = {"huggingface-hub": (0, 23), "numpy": (1, 26), "Pillow": (10, 0)}
ok = True
for pkg, min_ver in required.items():
    try:
        ver = tuple(int(x) for x in m.version(pkg).split(".")[:2])
        if ver < min_ver:
            print(f"  ✗ {pkg} {ver} < {min_ver}", file=sys.stderr)
            ok = False
        else:
            print(f"  ✓ {pkg} {ver}")
    except m.PackageNotFoundError:
        print(f"  ✗ {pkg} not found", file=sys.stderr)
        ok = False
if not ok:
    print("\nEnvironment check failed.", file=sys.stderr)
    exit(1)
EOF

            echo ""
            echo "  yolo26x download shell — ready"
            echo "  hf-cli login"
            echo "  hf-cli download HaiderKhan6410/weapon-yolo26x --local-dir model/"
            echo ""
          '';
        };

        # ── nix develop   (or  nix develop .#default) ─────────────────────────
        # Full dev shell: adds git, pre-commit, and a venv managed by Nix so
        # the pip-installed ML stack (torch, ultralytics, gradio …) lives in
        # .venv and is NOT under Nix control — that boundary is intentional.
        # Torch / TRT are too large and too GPU-specific to package in Nix
        # cleanly; pip inside a Nix-managed venv is the right trade-off.
        devShells.default = pkgs.mkShell {
          name     = "yolo26x-dev";
          packages = [
            (python.withPackages commonPythonPkgs)
            hf-cli
            pkgs.git
            pkgs.stdenv.cc.cc.lib   # libstdc++ for torch .so files
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.libGL               # OpenCV headless needs this on Linux
            pkgs.glib                # libglib-2.0
          ];

          # Make glibc / libstdc++ visible to pip-installed native extensions
          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.libGL
              pkgs.glib
            ];
          };

          shellHook = ''
            export HF_HOME="$PWD/.cache/huggingface"
            export PYTHONDONTWRITEBYTECODE=1
            export PYTHONUNBUFFERED=1

            # ── venv bootstrap ───────────────────────────────────────────────
            # We use a pip venv for the heavy ML stack (torch, ultralytics,
            # gradio) because nixpkgs doesn't package them reliably for all
            # GPU configs. The Nix shell provides the Python interpreter and
            # system libs; pip owns site-packages.
            VENV_DIR="$PWD/.venv"
            if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
              echo "→ Creating venv at .venv …"
              python -m venv "$VENV_DIR" --prompt yolo26x
              "$VENV_DIR/bin/pip" install --quiet --upgrade pip
              "$VENV_DIR/bin/pip" install --quiet -r requirements.txt
              echo "✓ venv ready"
            fi

            source "$VENV_DIR/bin/activate"

            # ── sanity checks ────────────────────────────────────────────────
            python - <<'EOF'
import sys, importlib.metadata as m, warnings
checks = [
    ("numpy",           (1, 26), (3, 0)),
    ("ultralytics",     (8,  3), (9, 0)),
    ("torch",           (2,  0), (4, 0)),
    ("gradio",          (4, 40), (6, 0)),
]
ok = True
for pkg, lo, hi in checks:
    try:
        ver = tuple(int(x) for x in m.version(pkg).split(".")[:2])
        if ver < lo or ver >= hi:
            print(f"  ✗ {pkg} {ver}  expected >={lo} <{hi}", file=sys.stderr)
            ok = False
        else:
            print(f"  ✓ {pkg} {ver}")
    except m.PackageNotFoundError:
        print(f"  ✗ {pkg} not found — run: pip install -r requirements.txt", file=sys.stderr)
        ok = False
if not ok:
    print("\nDependency check failed. Run: pip install -r requirements.txt", file=sys.stderr)
EOF

            echo ""
            echo "  yolo26x dev shell — ready"
            echo "  python main.py                  start the app"
            echo "  pytest tests/ -v                run tests"
            echo "  nix develop .#download          download-only shell"
            echo ""
          '';
        };

      }
    );
}

