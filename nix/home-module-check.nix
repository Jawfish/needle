{
  self,
  nixpkgs,
  system,
}:
let
  pkgs = import nixpkgs { inherit system; };
  inherit (pkgs) lib;
  isLinux = pkgs.stdenv.hostPlatform.isLinux;
  command = "${self.packages.${system}.needle}/bin/needle";
  toJson = value: builtins.unsafeDiscardStringContext (builtins.toJSON value);

  platformOptions =
    if isLinux then
      {
        options.systemd.user = {
          services = lib.mkOption {
            type = lib.types.attrsOf (lib.types.attrsOf lib.types.anything);
            default = { };
          };
          timers = lib.mkOption {
            type = lib.types.attrsOf (lib.types.attrsOf lib.types.anything);
            default = { };
          };
        };
      }
    else
      {
        options.home.homeDirectory = lib.mkOption {
          type = lib.types.str;
          default = "/Users/test";
        };
        options.launchd.agents = lib.mkOption {
          type = lib.types.attrsOf lib.types.anything;
          default = { };
        };
      };

  evaluate =
    mode: enable: serve:
    (lib.evalModules {
      specialArgs = { inherit pkgs; };
      modules = [
        self.homeModules.needle
        platformOptions
        {
          services.needle = {
            inherit enable mode;
            serve.enable = serve;
          };
        }
      ];
    }).config;

  units = configuration: if isLinux then configuration.systemd.user else configuration.launchd.agents;

  actual = {
    watch = units (evaluate "watch" true true);
    timer = units (evaluate "timer" true true);
    serveOnly = units (evaluate "watch" false true);
  };

  supervised = {
    Restart = "on-failure";
    RestartSec = 5;
    TimeoutStopSec = 120;
    Nice = 10;
    IOSchedulingClass = "idle";
    Environment = "NEEDLE_LOG=info";
  };

  systemdServe = {
    Unit.Description = "Needle document server";
    Service = supervised // {
      ExecStart = "${command} serve --host 127.0.0.1 --port 8080";
    };
    Install.WantedBy = [ "default.target" ];
  };

  expectedSystemd = {
    watch = {
      services = {
        needle-watch = {
          Unit.Description = "Needle document watcher";
          Service = supervised // {
            ExecStart = "${command} watch";
          };
          Install.WantedBy = [ "default.target" ];
        };
        needle-serve = systemdServe;
      };
      timers = { };
    };
    timer = {
      services = {
        needle-reindex = {
          Unit.Description = "Needle document reindex";
          Service = {
            Type = "oneshot";
            ExecStart = "${command} reindex";
            TimeoutStopSec = 120;
            Nice = 10;
            IOSchedulingClass = "idle";
            Environment = "NEEDLE_LOG=info";
          };
        };
        needle-serve = systemdServe;
      };
      timers.needle-reindex = {
        Unit.Description = "Needle periodic reindex";
        Timer = {
          OnBootSec = "15m";
          OnUnitActiveSec = "15m";
          Unit = "needle-reindex.service";
        };
        Install.WantedBy = [ "timers.target" ];
      };
    };
    serveOnly = {
      services.needle-serve = systemdServe;
      timers = { };
    };
  };

  launchdAgent = label: arguments: extra: {
    enable = true;
    config = {
      Label = label;
      ProgramArguments = [ command ] ++ arguments;
      ExitTimeOut = 120;
      ProcessType = "Background";
      EnvironmentVariables.NEEDLE_LOG = "info";
      StandardErrorPath = "/Users/test/Library/Logs/needle.log";
    }
    // extra;
  };

  launchdKeepAlive = {
    RunAtLoad = true;
    KeepAlive.SuccessfulExit = false;
    ThrottleInterval = 5;
  };

  launchdServe = launchdAgent "dev.needle.serve" [
    "serve"
    "--host"
    "127.0.0.1"
    "--port"
    "8080"
  ] launchdKeepAlive;

  expectedLaunchd = {
    watch = {
      needle-watch = launchdAgent "dev.needle.watch" [ "watch" ] launchdKeepAlive;
      needle-serve = launchdServe;
    };
    timer = {
      needle-reindex = launchdAgent "dev.needle.timer" [ "reindex" ] { StartInterval = 900; };
      needle-serve = launchdServe;
    };
    serveOnly.needle-serve = launchdServe;
  };

  expected = if isLinux then expectedSystemd else expectedLaunchd;
in
pkgs.runCommand "needle-home-module-check" { nativeBuildInputs = [ pkgs.jq ]; } ''
  jq -S . ${pkgs.writeText "expected.json" (toJson expected)} > expected.json
  jq -S . ${pkgs.writeText "actual.json" (toJson actual)} > actual.json
  diff -u expected.json actual.json
  touch $out
''
