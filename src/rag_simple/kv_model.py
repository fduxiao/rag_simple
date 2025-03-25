from copy import deepcopy
from pathlib import Path

import tomllib
import tomli_w


class Field:
    def __init__(self, name=None, default=None, default_factory=None):
        self.name = name
        self.default = default
        self.default_factory = default_factory

    def make_default(self):
        if self.default_factory is not None:
            return self.default_factory()
        return self.default

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        key = self.name
        data = instance.data
        if key in data:
            return data[key]
        # make default
        return self.make_default()

    def __set__(self, instance, value):
        instance.data[self.name] = value


class ModelField(Field):
    def __init__(self, model_class, name=None):
        super().__init__(name=name)
        self.model_class = model_class

    def make_default(self):
        return self.model_class.make_default()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        data = super().__get__(instance, owner)
        return self.model_class(data)

    def __set__(self, instance, value):
        if isinstance(value, self.model_class):
            value = value.data
        super().__set__(instance, value)


class KVModel:
    fields = {}

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.fields = {}
        for key, field in cls.__dict__.items():
            if isinstance(field, Field):
                cls.fields[key] = field

    @classmethod
    def as_field(cls, name=None):
        return ModelField(cls, name=name)

    @classmethod
    def make_default(cls):
        result = {}
        for key, field in cls.fields.items():
            result[key] = field.make_default()
        return result

    def __init__(self, data=None):
        if data is None:
            data = self.make_default()
        self.data = data

    def __getitem__(self, item):
        return self.data["item"]

    def __setitem__(self, key, value):
        self[key] = value

    def dump(self):
        return deepcopy(self.data)

    def load(self, data: dict):
        for key, value in data.items():
            field = self.fields[key]
            field.__set__(self, value)

    def from_toml(self, path):
        with open(path, 'rb') as file:
            data = tomllib.load(file)
            self.load(data)
        return self

    def from_config_file(self, path: str | Path, write_on_absence=False):
        path = Path(path)
        if not path.exists() and write_on_absence:
            path.parent.mkdir(exist_ok=True, parents=True)
            # TODO: check file extension
            self.to_toml(path)
        # TODO: check file extension
        return self.from_toml(path)

    def to_toml(self, path):
        with open(path, 'wb') as file:
            tomli_w.dump(self.data, file)
        return self
