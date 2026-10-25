import argparse
import enum
import os
import comfy.options


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


LOG_LEVELS = ('DEBUG', 'DETAIL', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')


class VerboseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 0:
            output = ('DEBUG', None)
        elif len(values) == 1 and values[0] in LOG_LEVELS:
            output = (values[0], None)
        elif len(values) == 2 and values[0] in LOG_LEVELS:
            output = tuple(values)
        else:
            parser.error(f"{option_string} expects no values, a console LEVEL, or LEVEL FILE")
        setattr(namespace, self.dest, [*getattr(namespace, self.dest, []), output])


def get_console_log_level(outputs):
    console_levels = [level for level, path in outputs if path is None]
    return min(console_levels, key=LOG_LEVELS.index, default='INFO')


def get_file_log_outputs(outputs):
    return [(level, path) for level, path in outputs if path is not None]


parser = argparse.ArgumentParser()

parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::", help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--tls-keyfile", type=str, help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function")
parser.add_argument("--tls-certfile", type=str, help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function")
parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")

parser.add_argument("--base-directory", type=str, default=None, help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")
parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append', help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory. Overrides --base-directory.")
parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory (default is in the ComfyUI directory). Overrides --base-directory.")
parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory. Overrides --base-directory.")
parser.add_argument("--previews-directory", type=str, default=None, help="Set the directory for generated asset previews. Overrides --base-directory.")
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        return None

parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum media preview size.")

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--cache-ram", nargs='*', type=float, default=[], metavar="GB", help="Use RAM pressure caching with the specified headroom thresholds. This is the default caching mode. The first value sets the active-cache threshold; the optional second value sets the inactive-cache/pin threshold. Defaults when no values are provided: active 10%% of system RAM (min 2GB, max 10GB), inactive 100%% of system RAM (max 128GB).")
cache_group.add_argument("--cache-classic", action="store_true", help="Use the old style (aggressive) caching.")
cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")
cache_group.add_argument("--cache-none", action="store_true", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")
cache_group.add_argument("--high-ram", action="store_true", help="Can improve performance slightly on high RAM or on systems where pagefile use is preferred over model loading.")

parser.add_argument("--enable-manager", action="store_true", help="Enable the ComfyUI-Manager feature.")
parser.add_argument("--enable-agent", action="store_true", help="Enable the ComfyUI agent feature.")
manager_group = parser.add_mutually_exclusive_group()
manager_group.add_argument("--disable-manager-ui", action="store_true", help="Disables only the ComfyUI-Manager UI and endpoints. Scheduled installations and similar background tasks will still operate.")
manager_group.add_argument("--enable-manager-legacy-ui", action="store_true", help="Enables the legacy UI of ComfyUI-Manager. Implies --enable-manager.")


parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")

parser.add_argument("--debug-hang", action="store_true", help="Enable stack trace dumps on Ctrl-C for debugging hangs.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--windows-standalone-build", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup and turning on the assets system; --disable-assets turns it off).")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")
parser.add_argument("--whitelist-custom-nodes", type=str, nargs='+', default=[], help="Use 'foundry' to load the bundled extension when --disable-all-custom-nodes is enabled. Other packages are not loaded.")
parser.add_argument("--disabled-nodes-config", type=str, default=None, metavar="PATH", help="Path to a YAML file listing node IDs to disable.")
parser.add_argument("--disable-partner-nodes", action="store_true", help="Disable the bundled Microsoft Foundry extension.")
parser.add_argument("--offline", action="store_true", help="Restrict frontend internet access and disable the Microsoft Foundry extension.")
parser.add_argument("--disable-api-nodes", action="store_true", help="Deprecated alias for --offline. Use --offline, or --disable-partner-nodes to only disable partner nodes.")

parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")

parser.add_argument("--verbose", action=VerboseAction, nargs='*', default=[], metavar='LEVEL FILE', help='Set console logging with no values or LEVEL, or add a LEVEL FILE log output. May be repeated.')
parser.add_argument("--log-stdout", action="store_true", help="Send normal process output to stdout instead of stderr (default).")


# The default built-in provider hosted under web/
DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"

parser.add_argument(
    "--front-end-version",
    type=str,
    default=DEFAULT_VERSION_STRING,
    help="""
    Specifies the version of the frontend to be used. This command needs internet connectivity to query and
    download available frontend implementations from GitHub releases.

    The version string should be in the format of:
    [repoOwner]/[repoName]@[version]
    where version is one of: "latest" or a valid version number (e.g. "1.0.0")
    """,
)

def is_valid_directory(path: str) -> str:
    """Validate if the given path is a directory, and check permissions."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory.")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"You do not have read permissions for '{path}'.")
    return path

parser.add_argument(
    "--front-end-root",
    type=is_valid_directory,
    default=None,
    help="The local filesystem path to the directory where the frontend is located. Overrides --front-end-version.",
)

parser.add_argument("--user-directory", type=is_valid_directory, default=None, help="Set the ComfyUI user directory with an absolute path. Overrides --base-directory.")
parser.add_argument("--models-directory", type=is_valid_directory, default=None, help="Set the ComfyUI models directory. Overrides the models folder in --base-directory.")

parser.add_argument("--enable-compress-response-body", action="store_true", help="Enable compressing response body.")

parser.add_argument(
    "--comfy-api-base",
    type=str,
    default="https://api.comfy.org",
    help="Set the base URL for the ComfyUI API.  (default: https://api.comfy.org)",
)

database_default_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "user", "comfyui.db")
)
parser.add_argument("--database-url", type=str, default=None, help="Specify the database URL, e.g. for an in-memory database you can use 'sqlite:///:memory:'. Defaults to 'comfyui.db' in the effective user directory.")
parser.add_argument("--enable-assets", action="store_true", help="Enable the assets system (API routes, database synchronization, and background scanning). Off by default.")
parser.add_argument("--disable-assets", action="store_true", help="Disable the assets system, even when --enable-assets is also given.")
parser.add_argument("--enable-asset-hashing", action="store_true", help="Compute blake3 content hashes when scanning assets. Files uploaded through the API are hashed either way. Hashing enables future asset-portability features (deduplication, cross-machine model resolution) but adds startup cost and per-output cost on large models directories. Off by default; enable to opt in. Requires the assets system (--enable-assets or --windows-standalone-build).")
parser.add_argument("--feature-flag", type=str, action='append', default=[], metavar="KEY[=VALUE]", help="Set a server feature flag. Use KEY=VALUE to set an explicit value, or bare KEY to set it to true. Can be specified multiple times. Boolean values (true/false) and numbers are auto-converted. Examples: --feature-flag show_signin_button=true  or  --feature-flag show_signin_button")
parser.add_argument("--list-feature-flags", action="store_true", help="Print the registry of known CLI-settable feature flags as JSON and exit.")

if comfy.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.cache_ram is not None and len(args.cache_ram) > 2:
    parser.error("--cache-ram accepts at most two values: active GB and inactive GB")

if args.high_ram:
    args.cache_classic = True

if args.windows_standalone_build:
    args.auto_launch = True
    args.enable_assets = True

if args.disable_api_nodes:
    args.offline = True

if args.offline:
    args.disable_partner_nodes = True

# Assets are off unless --enable-assets is given; --disable-assets wins over it.
if not args.enable_assets:
    args.disable_assets = True

if args.disable_auto_launch:
    args.auto_launch = False

# '--enable-manager-legacy-ui' is meaningless unless the manager is enabled, so imply '--enable-manager'.
if args.enable_manager_legacy_ui:
    args.enable_manager = True
