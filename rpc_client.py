from t50hz_rpc import rpc_client

from functions.parse import get_argparser, get_config

RPC_ENDPOINT = "rpc_online_outlier_detection"

if __name__ == '__main__':
    parser = get_argparser()
    parser.add_argument("-d", "--debug",
                        help="Debug the file using loop as source",
                        default=False, type=bool)
    args = parser.parse_args()

    config = get_config(args.config_file)

    client = rpc_client(RPC_ENDPOINT)

    client.start(config, args.topic, args.key_path, args.debug)
