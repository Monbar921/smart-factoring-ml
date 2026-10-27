"""Загрузка pickle моделей коллеги: в файлах классы записаны под именами модулей, в которых их обучали
(pd_model, pd_features, dl_model), а в сервисе эти модули лежат в пакетах app.models.pd / app.models.delay."""

import importlib
import pickle


class _Renaming(pickle.Unpickler):
    def __init__(self, file, modules: dict[str, str]):
        super().__init__(file)
        self.modules = modules

    def find_class(self, module, name):
        return super().find_class(self.modules.get(module, module), name)


def load(path: str, modules: dict[str, str]):
    for target in modules.values():
        importlib.import_module(target)
    with open(path, "rb") as file:
        return _Renaming(file, modules).load()
