{
  description = "Needle development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.cargo
              pkgs.clang
              pkgs.clippy
              pkgs.just
              pkgs.llvmPackages.libclang
              pkgs.openssl
              pkgs.pkg-config
              pkgs.rust-analyzer
              pkgs.rustc
              pkgs.rustfmt
            ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          };
        });
    };
}
