from snownlp import SnowNLP

class Analyzer:
    @staticmethod
    def analyze_comment(comment):
        try:
            s = SnowNLP(comment)            # 创建对象
            sentiment = s.sentiments            # 获取情感得分，范围在0到1之间

            if sentiment > 0.6:
                return {'sentiment': 'positive', 'score': float(sentiment)}            # 大于0.6为积极
            elif sentiment < 0.4:
                return {'sentiment': 'negative', 'score': float(sentiment)}            # 小于0.4为消极
            else:
                return {'sentiment': 'neutral', 'score': float(sentiment)}            # 其他情况为中性
        except Exception as e:
            print(f"情感分析失败: {str(e)}")
        return {'sentiment': 'neutral', 'score': 0.5}            # 返回默认的中性结果

    @staticmethod
    def analyze_comments(comments):
        results = []
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}        # 初始化情感统计字典
        for comment in comments:
            analysis = Analyzer.analyze_comment(comment)            # 对每条评论进行情感分析
            results.append({                                                           # 将分析结果添加到结果列表
                'text': comment,                                                       # 保存原始评论文本
                'sentiment': analysis['sentiment'],                         # 保存情感分类结果
                'score': analysis['score']})                # 保存情感得分
            sentiment_counts[analysis['sentiment']] += 1               # 更新情感统计计数
        return {'comments': results,
                'sentiment_counts': sentiment_counts}

