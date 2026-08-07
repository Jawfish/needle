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
      # Upstream prebuilt runtime. nixpkgs builds onnxruntime from source with
      # LTO and its full test suite, which costs an hour whenever the binary
      # cache misses.
      onnxruntimeFor =
        pkgs:
        let
          version = "1.27.1";
          releases = {
            x86_64-linux = {
              platform = "linux-x64";
              hash = "sha256-JbHvH+oazSENY/jyTchwrW4Hd5XOH1SHYlLG04A8Fa8=";
            };
            aarch64-linux = {
              platform = "linux-aarch64";
              hash = "sha256-M8Z+M9HiW4FoeDZuonZYmgJPcfAA5/+VXEszIk1jnt0=";
            };
            aarch64-darwin = {
              platform = "osx-arm64";
              hash = "sha256-5Ct3pygcxuVRQb9E/PusLHgrgjpJG7tqwzx4HdmR+KY=";
            };
          };
          release = releases.${pkgs.stdenv.hostPlatform.system} or null;
        in
        if release == null then
          pkgs.onnxruntime
        else
          pkgs.stdenv.mkDerivation {
            pname = "onnxruntime-bin";
            inherit version;
            src = pkgs.fetchurl {
              url = "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-${release.platform}-${version}.tgz";
              inherit (release) hash;
            };
            nativeBuildInputs = pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.autoPatchelfHook;
            buildInputs = [ pkgs.stdenv.cc.cc.lib ];
            dontBuild = true;
            installPhase = ''
              runHook preInstall
              mkdir -p "$out"
              cp -r include lib "$out/"
              runHook postInstall
            '';
            meta = {
              inherit (pkgs.onnxruntime.meta) description homepage;
              license = pkgs.lib.licenses.mit;
              platforms = builtins.attrNames releases;
            };
          };
      # Release binaries statically link the runtime, so installing one costs a
      # download rather than a Rust build.
      needleBinFor =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          version = "0.6.0";
          releases = {
            x86_64-linux = {
              platform = "linux-x86_64";
              hash = "sha256-zjK1IltD219ney3p0aHf4w9JsEzjtQ1TLXKoqGXGlCA=";
            };
            aarch64-darwin = {
              platform = "macos-aarch64";
              hash = "sha256-CDsC6PdBT9cF8qUwJK9aQ/PXJd/62pJnbkeUg9jnnbk=";
            };
          };
          release = releases.${system} or null;
        in
        if release == null then
          needleFor system
        else
          pkgs.stdenv.mkDerivation {
            pname = "needle";
            inherit version;
            src = pkgs.fetchurl {
              url = "https://github.com/Jawfish/needle/releases/download/v${version}/needle-v${version}-${release.platform}.tar.gz";
              inherit (release) hash;
            };
            sourceRoot = ".";
            nativeBuildInputs = pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.autoPatchelfHook;
            buildInputs = [
              pkgs.stdenv.cc.cc.lib
              pkgs.openssl
            ];
            dontBuild = true;
            installPhase = ''
              runHook preInstall
              install -Dm755 needle "$out/bin/needle"
              runHook postInstall
            '';
            meta = {
              description = "Local semantic search for documents";
              homepage = "https://github.com/Jawfish/needle";
              license = pkgs.lib.licenses.mit;
              mainProgram = "needle";
              platforms = builtins.attrNames releases;
            };
          };
      needleFor =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          onnxruntime = onnxruntimeFor pkgs;
        in
        pkgs.rustPlatform.buildRustPackage {
          pname = "needle";
          version = "0.6.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            outputHashes = {
              "xberg-1.0.14" = "sha256-dsGUboWzcU8F+06tVVi760607bx9YWwZRevu1Ik2/A8=";
            };
          };
          nativeBuildInputs = [ pkgs.pkg-config ];
          nativeCheckInputs = [ pkgs.cacert ];
          buildInputs = [
            onnxruntime
            pkgs.openssl
          ];
          ORT_LIB_LOCATION = "${onnxruntime}/lib";
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
      homeModules.needle = import ./nix/home-module.nix { inherit self; };
      homeManagerModules.needle = self.homeModules.needle;

      checks = forAllSystems (system: {
        home-module = import ./nix/home-module-check.nix {
          inherit self nixpkgs system;
        };
      });

      packages = forAllSystems (
        system:
        let
          needle = needleFor system;
        in
        {
          inherit needle;
          needle-bin = needleBinFor system;
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
