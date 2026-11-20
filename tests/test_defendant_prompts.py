"""Check single/multiple defendant prompts without loading inference models."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SINGLE = '得到和本案最相关的一个法条和罪名预测结果。只关注张某的犯罪行为，把你的最终答案放在方括号【】中，格式：本案的相关法条是：第*条，本案的罪名是：*罪。'
MULTI = '为张某,李某中每个罪犯预测相关法条和罪名。\n输出格式为:【罪犯1:法条:……,罪名:……】【罪犯2:……】……'
SINGLE_CASE = {'fact': '案情', 'meta': {'criminals': ['张某'], 'accusation': ['盗窃'], 'relevant_articles': [264]}}
MULTI_CASE = {'fact': '案情', 'defendants': ['张某', '李某'], 'outcomes': [
    {'judgment': [{'standard_accusation': '盗窃', 'article': ['264']}]},
    {'judgment': [{'standard_accusation': '诈骗', 'article': ['266']}]}]}


def function(path, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class DefendantPromptTests(unittest.TestCase):
    def test_direct(self):
        prompt = function('direct_reasoning/llm.py', 'generate_prompt')
        prefix = '你要做的任务是相关法条预测和罪名预测。\n本案事实：\n案情\n请分析当前案件，根据你自己的知识，'
        self.assertEqual(prompt('案情', '张某', False), prefix + SINGLE)
        self.assertEqual(prompt('案情', '张某,李某', True), prefix + MULTI)

    def test_rag_with_both_precedent_schemas(self):
        prompt = function('retrieval_augmented_reasoning/rag.py', 'retrieved_label_option_fewshot',
                          {'case_list': [SINGLE_CASE, MULTI_CASE]})
        for multi, criminal in [(False, '张某'), (True, '张某,李某')]:
            output = prompt('案情', criminal, [1, 2], multi)
            self.assertTrue(output.endswith(MULTI if multi else SINGLE))
            self.assertIn('本案的相关法条是：第264条，本案的罪名是：盗窃。', output)
            self.assertIn('罪犯李某: 法条是: 266；罪名是: 诈骗。', output)

    def test_glare_all_models(self):
        prompt = function('glare/prompts.py', 'get_search_agent_user_prompt')
        for model in ['qwq', '7b', '32b', '72b']:
            single = prompt('案情', '张某', model, False)
            multi = prompt('案情', '张某,李某', model, True)
            self.assertIn(SINGLE + '\n你可以按照以下步骤进行：', single)
            self.assertIn(MULTI + '你可以按照以下步骤进行：', multi)
            # All reasoning steps after the target/output instruction must match.
            self.assertEqual(single.split('你可以按照以下步骤进行：')[1],
                             multi.split('你可以按照以下步骤进行：')[1])

    def test_case_field_selection_in_all_three_entrypoints(self):
        for path, container in [('direct_reasoning/llm.py', 'case'),
                                ('retrieval_augmented_reasoning/rag.py', 'case'),
                                ('glare/glare.py', 'item')]:
            tree = ast.parse((ROOT / path).read_text())
            branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                          and isinstance(n.test, ast.Name) and n.test.id == 'is_multi_defendant'
                          and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'criminal' for t in x.targets) for x in n.body))
            for case, expected in [(SINGLE_CASE, '张某'), (MULTI_CASE, '张某,李某')]:
                ns = {container: case, 'is_multi_defendant': 'defendants' in case}
                exec(compile(ast.Module(body=[branch], type_ignores=[]), path, 'exec'), ns)
                self.assertEqual(ns['criminal'], expected)


if __name__ == '__main__':
    unittest.main()
