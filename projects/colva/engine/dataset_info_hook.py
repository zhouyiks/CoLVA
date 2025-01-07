from xtuner.engine.hooks import DatasetInfoHook
from ..dataset.utils import VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN

class DatasetInfoHook_withSpecialTokens(DatasetInfoHook):
    def __init__(self, tokenizer, is_intern_repo_dataset=False):
        super(DatasetInfoHook_withSpecialTokens, self).__init__(tokenizer, is_intern_repo_dataset)

        self._add_special_tokens()
    
    def _add_special_tokens(self):
        special_tokens = [VPT_CONTEXT_TOKEN,]
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)