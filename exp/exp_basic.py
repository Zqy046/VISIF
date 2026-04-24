from models import AutoTimes_Llama, AutoTimes_Gpt2, AutoTimes_Opt_1b, AutoTimes_InternVL, AutoTimes_InternLM


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'AutoTimes_Llama': AutoTimes_Llama,
            'AutoTimes_Gpt2': AutoTimes_Gpt2,
            'AutoTimes_Opt_1b': AutoTimes_Opt_1b,
            'AutoTimes_InternVL': AutoTimes_InternVL,
            'AutoTimes_InternLM': AutoTimes_InternLM
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
