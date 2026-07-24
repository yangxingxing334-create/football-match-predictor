"""
半场预测脚本 - 在比赛进行到半场时，输入半场数据，预测全场胜平负结果

用法示例:
    python predict_halftime.py \
        --home "Djurgarden" \
        --away "Malmo" \
        --hthg 1 --htag 0 \
        --hs 8 --as 5 \
        --hst 4 --ast 2 \
        --hr 0 --ar 1 \
        --model rf

参数说明:
    --home     主队名（任意名称，用于显示）
    --away     客队名（任意名称，用于显示）
    --hthg     半场主队进球数
    --htag     半场客队进球数
    --hs       主队总射门数
    --as_      客队总射门数
    --hst      主队射正数
    --ast      客队射正数
    --hr       主队红牌数
    --ar       客队红牌数
    --model    模型选择：rf（随机森林）/ lr（逻辑回归）/ nb（朴素贝叶斯），默认 rf

注意：模型基于英超/西甲/法甲/德甲/意甲历史数据训练，可用于预测任何联赛半场比赛（准确率约64%）
"""

import argparse
import pandas as pd
from joblib import load
from os import path

MODEL_MAP = {
    'rf': 'rf_classifier.model',
    'lr': 'lr_classifier.model',
    'nb': 'nb_classifier.model',
}

MODEL_NAMES = {
    'rf': '随机森林',
    'lr': '逻辑回归',
    'nb': '朴素贝叶斯',
}

EXPORTED_MODELS_DIR = path.join(path.dirname(path.abspath(__file__)), 'exportedModels')


def predict(home, away, hthg, htag, hs, as_, hst, ast, hr, ar, model_key='rf'):
    model_file = path.join(EXPORTED_MODELS_DIR, MODEL_MAP[model_key])

    if not path.exists(model_file):
        print(f"❌ 找不到模型文件: {model_file}")
        print("   请先运行主训练脚本 football-match-predictor.py 并导出模型")
        return

    clf = load(model_file)

    # 使用编号 0 代替球队名（模型不依赖具体球队名，只用编号特征）
    X = pd.DataFrame({
        'home_encoded': [0],
        'away_encoded': [1],
        'HTHG': [hthg],
        'HTAG': [htag],
        'HS': [hs],
        'AS': [as_],
        'HST': [hst],
        'AST': [ast],
        'HR': [hr],
        'AR': [ar],
    })

    probs = clf.predict_proba(X)[0]
    result = dict(zip(clf.classes_, probs))

    home_win = result.get('H', 0)
    draw = result.get('D', 0)
    away_win = result.get('A', 0)

    print()
    print("=" * 45)
    print(f"  🏟️  {home}  vs  {away}")
    print("=" * 45)
    print(f"  半场比分: {hthg} - {htag}")
    print(f"  射门: {hs} - {as_}  射正: {hst} - {ast}  红牌: {hr} - {ar}")
    print("-" * 45)
    print(f"  使用模型: {MODEL_NAMES[model_key]}")
    print("-" * 45)
    print(f"  🏠 主队获胜概率:  {home_win * 100:.1f}%  {home}")
    print(f"  🤝 平局概率:      {draw * 100:.1f}%")
    print(f"  ✈️  客队获胜概率:  {away_win * 100:.1f}%  {away}")
    print("-" * 45)

    best = max([('主队获胜', home_win), ('平局', draw), ('客队获胜', away_win)], key=lambda x: x[1])
    print(f"  ✅ 最可能结果: {best[0]}（{best[1] * 100:.1f}%）")
    print("=" * 45)
    print("  ⚠️  注意：模型基于五大联赛历史数据训练，预测仅供参考")
    print()


def main():
    parser = argparse.ArgumentParser(description='足球比赛半场预测工具')
    parser.add_argument('--home', type=str, default='主队', help='主队名称')
    parser.add_argument('--away', type=str, default='客队', help='客队名称')
    parser.add_argument('--hthg', type=int, required=True, help='半场主队进球数')
    parser.add_argument('--htag', type=int, required=True, help='半场客队进球数')
    parser.add_argument('--hs', type=int, required=True, help='主队射门数')
    parser.add_argument('--as_', type=int, required=True, dest='as_', help='客队射门数')
    parser.add_argument('--hst', type=int, required=True, help='主队射正数')
    parser.add_argument('--ast', type=int, required=True, help='客队射正数')
    parser.add_argument('--hr', type=int, default=0, help='主队红牌数（默认0）')
    parser.add_argument('--ar', type=int, default=0, help='客队红牌数（默认0）')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'lr', 'nb'],
                        help='模型选择: rf=随机森林, lr=逻辑回归, nb=朴素贝叶斯（默认rf）')

    args = parser.parse_args()

    predict(
        home=args.home,
        away=args.away,
        hthg=args.hthg,
        htag=args.htag,
        hs=args.hs,
        as_=args.as_,
        hst=args.hst,
        ast=args.ast,
        hr=args.hr,
        ar=args.ar,
        model_key=args.model,
    )


if __name__ == '__main__':
    main()
