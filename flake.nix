{
  description = "Needle development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      needleFor =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.rustPlatform.buildRustPackage {
          pname = "needle";
          version = "0.4.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            outputHashes = {
              "xberg-1.0.14" = "sha256-9WINAtQs74zRpCWGjB6lLDZDXIsXlIBLyqQJ+C0wKLU=";
            };
          };
          nativeBuildInputs = [ pkgs.pkg-config ];
          nativeCheckInputs = [ pkgs.cacert ];
          buildInputs = [
            pkgs.onnxruntime
            pkgs.openssl
          ];
          ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";
          preCheck = ''
            export HOME="$TMPDIR/home"
            export XDG_CONFIG_HOME="$HOME/config"
            export XDG_DATA_HOME="$HOME/data"
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
          '';
          meta = {
            description = "Local semantic search for documents";
            homepage = "https://github.com/Jawfish/needle";
            license = pkgs.lib.licenses.mit;
            mainProgram = "needle";
          };
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          needle = needleFor system;
        in
        {
          inherit needle;
          default = needle;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.needle}/bin/needle";
          meta.description = "Local semantic search for documents";
        };
      });

      devShells = forAllSystems (
        system:
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
        }
      );
    };
}
