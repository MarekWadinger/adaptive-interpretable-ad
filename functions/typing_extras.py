from typing import Union
from typing_extensions import TypedDict, NotRequired

from pandas import Timedelta


class FileClient(TypedDict):
    path: str
    output: str


class MQTTClient(TypedDict):
    host: str
    port: int


class KafkaClient(TypedDict):
    bootstrap_servers: str


class PulsarClient(TypedDict):
    service_url: str


class IOConfig(TypedDict):
    in_topics: list[str]
    out_topics: Union[list[str], str, None]


class ModelConfig(TypedDict):
    threshold: float
    t_e: Timedelta
    t_a: Union[Timedelta, None]
    t_g: Union[Timedelta, None]


class SetupConfig(TypedDict):
    recovery_path: NotRequired[str]
    key_path: NotRequired[str]
    debug: NotRequired[bool]


def istypedinstance(obj, type_):
    for property_name, property_type in type_.__annotations__.items():
        value = obj.get(property_name, None)
        if value is None:
            return False
        elif property_type not in (int, float, bool, str):
            # check if property_type is object (e.g. not a primitive)
            result = isinstance(property_type, value)
            if result is False:
                return False
        elif not isinstance(value, property_type):
            # Check for type equality
            return False
    return True
