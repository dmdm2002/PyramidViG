from RunModules import trainer
from Options import param


class driver(param):
    def __init__(self):
        super(driver, self).__init__()

    def run_train(self):
        tr = trainer.train()
        tr.run()

    def __call__(self, *args, **kwargs):
        if self.run_type == 0:
            return self.run_train()


if __name__ == "__main__":
    driver()()