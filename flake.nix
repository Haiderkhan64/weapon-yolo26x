{
  description = "YOLO26x-v1 dev environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pythonEnv = pkgs.python313.withPackages (ps: with ps; [
        huggingface-hub
        pip
        numpy
        pillow
        tqdm
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pythonEnv ];

        shellHook = ''
          export PATH="${pythonEnv}/bin:$PATH"
          export HF_HOME="$PWD/.cache/huggingface"
          echo "✓ huggingface-cli ready"
        '';
      };
    };
}
