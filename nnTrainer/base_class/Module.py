from .. import Configurator

class Module():
    def __init__(self) -> None:
        self.config = Configurator()

    def before(self, step: str):
        pass

    def onRunning(self):
        pass

    def after(self, step: str):
        pass