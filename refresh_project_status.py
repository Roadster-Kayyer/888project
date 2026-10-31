"""Summarize completed computation and unresolved research work from current files."""

import json
from pathlib import Path

import pandas as pd

from project_support import ROOT
from review_audit import audit_reformulations


def main():
    _, audit = audit_reformulations()
    audit_faqs = pd.read_csv(ROOT / 'retrieval_supplement' / 'query_quality_audit_100.csv', dtype={'doc_id': str}).doc_id.nunique()
    proposals = [
        ('reformulation', 'Q006714', 'meaning_preserved_yes_no_uncertain', 'yes', 'no or uncertain', '匿名要求被弱化；现有备注与保留原意=yes不一致。'),
        ('reformulation', 'Q008885', 'meaning_preserved_yes_no_uncertain', 'yes', 'no or uncertain', '法律后果被缩小为罚款/禁止；请与原问题范围对照。'),
        ('rag_formal', 'R00519', 'human_faithfulness', '2', '1 (researcher decision)', '把性侵处罚推及体育赛事暴力，证据未建立这种适用关系。'),
        ('rag_policybench_formal_v3', 'R00791', 'human_faithfulness', '2', 'blank / N/A', '仅说明证据缺少限额，按既定纯拒答规则应留空；请研究者确认。'),
        ('rag_policybench_formal_v3', 'R00333', 'human_faithfulness', '2', 'reassess', '回答猜测口号含义；Document 5只说明资料日期，不能支持该猜测。')]
    rows = []
    for dataset, identifier, field, old, suggested, reason in proposals:
        path = ROOT / 'retrieval_supplement' / 'query_quality_audit_100.csv' if dataset == 'reformulation' else ROOT / dataset / 'human_review_blinded.csv'
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
        key = 'query_id' if dataset == 'reformulation' else 'review_id'
        current = frame.loc[frame[key].eq(identifier), field].iloc[0]
        if current == old:
            rows.append({'dataset': dataset, 'id': identifier, 'field': field, 'researcher_value': current,
                         'assistant_suggestion': suggested, 'reason': reason,
                         'provenance': 'AI-assisted suggestion; not applied to independent researcher labels'})
    pd.DataFrame(rows, columns=['dataset', 'id', 'field', 'researcher_value', 'assistant_suggestion', 'reason', 'provenance']).to_csv(
        ROOT / 'provenance' / 'researcher_confirmation_needed.csv', index=False, encoding='utf-8-sig')
    sections = ['# 项目当前状态', '', '四份笔记本使用已保存的固定数据与生成记录进行复核。计算测试通过不等于所有研究结论已经得到人工或来源验证。', '',
                f"改写审核：{audit['n_completed_labels']}/{audit['n_sample']} 条标签已填写，涉及 {audit_faqs} 个 FAQ，每个 FAQ 四种改写；{audit['n_issues']} 个一致性提示。原始研究者标签未被维护脚本修改。"]
    for name in ['rag_formal', 'rag_policybench_formal_v3']:
        directory = ROOT / name
        evaluation = pd.read_csv(directory / 'answer_evaluation.csv')
        reviews = pd.read_csv(directory / 'human_review_blinded.csv', keep_default_na=False)
        calibration = pd.read_csv(directory / 'judge_human_calibration.csv')
        flags = evaluation[['query_id', 'system', 'query_type']].copy()
        flags['retrieval_miss'] = evaluation.system.ne('No-RAG') & evaluation.strict_gold_in_top5.eq(False)
        flags['invalid_citation_detected'] = evaluation.has_invalid_citation
        flags['low_judge_correctness'] = evaluation.correctness.lt(2)
        flags['output_length_limit'] = evaluation.hit_token_limit
        flags['interpretation'] = 'Automatically derived flags; not researcher-assigned error categories.'
        flags.to_csv(directory / 'automated_error_flags.csv', index=False, encoding='utf-8-sig')
        sections += ['', f'## {name}', f"回答 {len(evaluation)}，格式有效评分 {int(evaluation.judge_status.eq('ok').sum())}，人工正确性已填 {int(reviews.human_correctness.ne('').sum())} 条。",
                     '', '```', calibration.to_string(index=False), '```',
                     '主表分数仍是0–2序数评分，不应写成客观准确率。校准保留研究者原始标签；适用性分歧见 human_review_issues.csv。人工备注和错误分类须按真实检查补写，自动标记不可冒充人工分析。']
    source = pd.read_csv(ROOT / 'policybench_data' / 'source_quality_audit.csv', keep_default_na=False)
    sections += ['', '## 尚待研究者确认', f'具体评分/标签建议 {len(rows)} 条，见 provenance/researcher_confirmation_needed.csv；它们是助手建议，未冒充独立人工评分。',
                 f"PolicyBench来源表：{len(source)} 行，已填URL {int(source.source_url.ne('').sum())} 行，研究者确认 {int(source.reviewed_by_researcher.ne('').sum())} 行。若仅作为公开FAQ答案基准，应披露尚未逐条核验；若主张当前政策事实或官方原文检索，须提供对应来源证据。",
                 '实验三人工样本有20题，每题四系统。领域题数以 review_sample_metadata.json 为准；选择机制尚未记录，不能声称每领域等量或事先随机抽样。',
                 '两个检索语料都使用参考FAQ答案。应说明这是答案语料基准，不是原始政府文件的端到端检索。新西兰特定结论需要另有新西兰证据。',
                 '整理代表性检索失误、生成偏题、无依据推断、引用错误和拒答案例，完成论文的结果、讨论与局限性。']
    (ROOT / 'PROJECT_STATUS.md').write_text('\n'.join(sections) + '\n', encoding='utf-8')
    print('Project status refreshed; researcher annotations preserved.')


if __name__ == '__main__':
    main()
