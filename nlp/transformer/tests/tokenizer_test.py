import unittest

from transformer.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_file = "/home/orient/chi/model/chinese_wwm_ext/vocab.txt"
        cls.tokenizer = Tokenizer(vocab_file=cls.vocab_file, do_lower_case=True)
        # cls.tokenizer = FullTokenizer(vocab_file=cls.vocab_file, do_lower_case=True)
        # cls.bert_token = BertTokenizer.from_pretrained("/home/orient/chi/model/chinese_wwm_ext")

    @classmethod
    def tearDownClass(cls):
        pass

    def testToken(self):
        text_a = "你好啊"
        text_b = "你也是"
        tokens = []
        tokens.append("[CLS]")
        tokens.extend(self.tokenizer.tokenize(text_a))
        tokens.append("[SEP]")
        tokens.extend(self.tokenizer.tokenize(text_b))
        tokens.append("[SEP]")

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(ids, [101, 872, 1962, 1557, 102, 872, 738, 3221, 102])

        ids_to_tokens = self.tokenizer.convert_ids_to_tokens(ids)
        self.assertEqual(ids_to_tokens, tokens)

    def testEncodeAndDecode(self):
        text_a = "你好啊"
        text_b = "你也是"
        input_ids, segments_ids = self.tokenizer.encode(
            first_text=text_a, second_text=text_b, max_length=16
        )
        self.assertEqual(input_ids, [101, 872, 1962, 1557, 102, 872, 738, 3221, 102])
        self.assertEqual(segments_ids, [0, 0, 0, 0, 0, 1, 1, 1, 1])
        text = self.tokenizer.decode(input_ids)
        self.assertEqual(text, text_a + text_b)
