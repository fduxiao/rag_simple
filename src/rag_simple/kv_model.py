from copy import deepcopy


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
        owner.fields[self.name] = self

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
        cls.fields = cls.fields.copy()
        KVModel.fields = {}

    @classmethod
    def as_field(cls):
        return ModelField(cls)

    @classmethod
    def make_default(cls):
        result = {}
        for key, field in cls.fields.items():
            result[key] = field.make_default()
        return result

    def __init__(self, data=None, name=None):
        if data is None:
            data = self.make_default()
        self.data = data
        self.name = name

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
