from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from typing import Union
from typing_extensions import TypedDict, NotRequired

from pandas import Timedelta

from functions.typing_extras import *

class Config(TypedDict):
    setup: SetupConfig
    model: ModelConfig
    io: IOConfig
    file: NotRequired[FileClient]
    mqtt: NotRequired[MQTTClient]
    kafka: NotRequired[KafkaClient]
    pulsar: NotRequired[PulsarClient]
    client: Union[FileClient, MQTTClient, KafkaClient, PulsarClient, None]

def update_dict_if_none(dict1, dict2):
    for key, value in dict2.items():
        if dict1.get(key) is None:
            dict1[key] = value


def get_params(
    ) -> Config:  
    """
    Parse command-line arguments and return the parsed namespace object.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser()

    setup_arg_grp = parser.add_argument_group('setup', 'setup related parameters')
    setup_arg_grp.add_argument(
        '-f', '--config-file', type=FileType('r'),
        default='config.ini')
    setup_arg_grp.add_argument('-r', '--recovery-path',
                        help='Path to store recovery models')
    setup_arg_grp.add_argument('-k', '--key-path', help='Path to RSA keys')
    setup_arg_grp.add_argument("-d", "--debug",
                        help="Debug the file using loop as source",
                        default=False, type=bool)

    model_arg_grp = parser.add_argument_group('model', 'Model related parameters')
    model_arg_grp.add_argument('--threshold', type=float)
    model_arg_grp.add_argument('--t-e', type=Timedelta)
    model_arg_grp.add_argument('--t-a', type=Timedelta)
    model_arg_grp.add_argument('--t-g', type=Timedelta)
    
    # client_arg_grp = parser.add_argument_group('client', 'Client related parameters')
    io_arg_grp = parser.add_argument_group('io')
    io_arg_grp.add_argument("-t", "--in-topics", nargs='*', type=str,
                        help="Topic of MQTT or Column of pd.DataFrame")
    io_arg_grp.add_argument('--out-topics', nargs='*', type=str)

    file_arg_grp = parser.add_argument_group('file client', "File source related parameters")
    file_arg_grp.add_argument('--path', type=str)
    file_arg_grp.add_argument('--output', type=int)

    mqtt_arg_grp = parser.add_argument_group('mqtt client', "MQTT source related parameters")
    mqtt_arg_grp.add_argument('--host', type=str)
    mqtt_arg_grp.add_argument('--port', type=int)
    
    kafka_arg_grp = parser.add_argument_group('kafka client', "Kafka source related parameters")
    kafka_arg_grp.add_argument('--bootstrap-servers', type=str)
    
    pulsar_arg_grp = parser.add_argument_group('pulsar client', "Pulsar source related parameters")
    pulsar_arg_grp.add_argument('--service-url', type=str)
    
    args = parser.parse_args()

    config: Config = {
        "setup": {"recovery_path": args.recovery_path,
                  "key_path": args.key_path,
                  "debug": args.debug},
        "model": {"threshold": args.threshold,
                    "t_e": args.t_e,
                    "t_a": args.t_a,
                    "t_g": args.t_g},
        "io": {"in_topics": args.in_topics,
                    "out_topics": args.out_topics},
        "file": {"path": args.path,
                    "output": args.output},
        "mqtt": {"host": args.host,
                    "port": args.port},
        "kafka": {"bootstrap_servers": args.bootstrap_servers},
        "pulsar": {"service_url": args.service_url},
        "client": None
    }
    
    config_struct = {
        "setup": SetupConfig,
        "model": ModelConfig,
        "io": IOConfig,
        "file": FileClient,
        "mqtt": MQTTClient,
        "kafka": KafkaClient,
        "pulsar": PulsarClient,
    }

    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)

    def make_valid_type(type_) -> Union[type, None]:
        if isinstance(type_, type):
            return type_
        elif hasattr(type_, "__args__"):
            # if any([t is type(None) for t in type_.__args__]):
            #     return type(None)
            if "NotRequired" in str(type_):
                return str
            else:
                return make_valid_type(type_.__args__[0])
        else:
            raise ValueError(f"Invalid type: {type_}")

    args_ = vars(args)
    config = {}
    for section, struct in config_struct.items():
        config[section] = {}
        for param, type_ in struct.__annotations__.items():
            type_ = make_valid_type(type_)
            if args_.get(param) is not None:
                param_value = args_[param]
            elif config_parser.has_option(section, param):
                param_value = config_parser[section][param]
            else:
                param_value = None

            if (param_value is not None and param_value != "None" and param_value != ""):
                config[section][param] = type_(param_value)
            elif (param_value is None or param_value == "None" or param_value == ""):
                config[section][param] = None

    # Raise an error if multiple or no clients are specified
    active_clients = []
    clients = ["file", "mqtt", "kafka", "pulsar"]
    for client in clients:
        missing_args = any([arg == None for arg in config[client].values()])
        if missing_args:
            del config[client]
        else:
            active_clients.append(client)
    if len(active_clients) > 1:
        raise ValueError(f"Multiple clients specified: {active_clients}")
    if len(active_clients) == 0:
        raise ValueError(f"Specify one of the clients: {clients}")
    if len(active_clients) == 1:
        config["client"] = config.pop(active_clients[0])

    return config
