{ self }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  inherit (lib)
    mkEnableOption
    mkIf
    mkMerge
    mkOption
    types
    ;

  cfg = config.services.needle;
  command = "${cfg.package}/bin/needle";
  intervalParts = builtins.match "([1-9][0-9]*)([smh])" cfg.interval;
  intervalSeconds =
    (builtins.fromJSON (builtins.elemAt intervalParts 0))
    * {
      s = 1;
      m = 60;
      h = 3600;
    }
    .${builtins.elemAt intervalParts 1};

  serviceSettings = {
    Restart = "on-failure";
    RestartSec = 5;
    TimeoutStopSec = 120;
    Nice = 10;
    IOSchedulingClass = "idle";
    Environment = "NEEDLE_LOG=${cfg.logLevel}";
  };

  watchService = {
    Unit.Description = "Needle document watcher";
    Service = serviceSettings // {
      ExecStart = "${command} watch";
    };
    Install.WantedBy = [ "default.target" ];
  };

  serveService = {
    Unit.Description = "Needle document server";
    Service = serviceSettings // {
      ExecStart = "${command} serve --host ${cfg.serve.host} --port ${toString cfg.serve.port}";
    };
    Install.WantedBy = [ "default.target" ];
  };

  reindexService = {
    Unit.Description = "Needle document reindex";
    Service = {
      Type = "oneshot";
      ExecStart = "${command} reindex";
      TimeoutStopSec = 120;
      Nice = 10;
      IOSchedulingClass = "idle";
      Environment = "NEEDLE_LOG=${cfg.logLevel}";
    };
  };

  reindexTimer = {
    Unit.Description = "Needle periodic reindex";
    Timer = {
      OnBootSec = cfg.interval;
      OnUnitActiveSec = cfg.interval;
      Unit = "needle-reindex.service";
    };
    Install.WantedBy = [ "timers.target" ];
  };

  launchdAgent = label: arguments: keepAlive: schedule: {
    enable = true;
    config = {
      Label = label;
      ProgramArguments = [ command ] ++ arguments;
      ExitTimeOut = 120;
      ProcessType = "Background";
      EnvironmentVariables.NEEDLE_LOG = cfg.logLevel;
      StandardErrorPath = "${config.home.homeDirectory}/Library/Logs/needle.log";
    }
    // keepAlive
    // schedule;
  };

  launchdKeepAlive = {
    RunAtLoad = true;
    KeepAlive.SuccessfulExit = false;
    ThrottleInterval = 5;
  };
in
{
  options.services.needle = {
    enable = mkEnableOption "Needle document indexing";

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.needle;
      defaultText = lib.literalExpression "self.packages.\${pkgs.stdenv.hostPlatform.system}.needle";
      description = "The Needle package to run.";
    };

    mode = mkOption {
      type = types.enum [
        "watch"
        "timer"
      ];
      default = "watch";
      description = "Whether to watch for changes or periodically reindex.";
    };

    interval = mkOption {
      type = types.strMatching "[1-9][0-9]*[smh]";
      default = "15m";
      description = "The periodic reindex interval.";
    };

    logLevel = mkOption {
      type = types.str;
      default = "info";
      description = "The value of NEEDLE_LOG.";
    };

    serve = {
      enable = mkEnableOption "Needle's browser interface";

      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "The address for Needle's browser interface.";
      };

      port = mkOption {
        type = types.port;
        default = 8080;
        description = "The port for Needle's browser interface.";
      };
    };
  };

  config =
    if pkgs.stdenv.hostPlatform.isDarwin then
      mkMerge [
        (mkIf (cfg.enable && cfg.mode == "watch") {
          launchd.agents.needle-watch = launchdAgent "dev.needle.watch" [ "watch" ] launchdKeepAlive { };
        })
        (mkIf (cfg.enable && cfg.mode == "timer") {
          launchd.agents.needle-reindex = launchdAgent "dev.needle.timer" [ "reindex" ] { } {
            StartInterval = intervalSeconds;
          };
        })
        (mkIf cfg.serve.enable {
          launchd.agents.needle-serve = launchdAgent "dev.needle.serve" [
            "serve"
            "--host"
            cfg.serve.host
            "--port"
            (toString cfg.serve.port)
          ] launchdKeepAlive { };
        })
      ]
    else
      mkMerge [
        (mkIf (cfg.enable && cfg.mode == "watch") {
          systemd.user.services.needle-watch = watchService;
        })
        (mkIf (cfg.enable && cfg.mode == "timer") {
          systemd.user.services.needle-reindex = reindexService;
          systemd.user.timers.needle-reindex = reindexTimer;
        })
        (mkIf cfg.serve.enable {
          systemd.user.services.needle-serve = serveService;
        })
      ];
}
