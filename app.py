from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from sentiment_analysis import Analyzer

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 使用随机生成的密钥

# 电影
movies_df = pd.read_csv('data/top250movies.csv')
# 评论
comments_df = pd.read_csv('data/all_comments.csv')

@app.route('/')
def home():
    return redirect(url_for('index'))

# 首页
@app.route('/index')
def index():
    try:
        # 将DataFrame转换为列表格式，与原数据结构保持一致
        datalist = []
        for _, row in movies_df.iterrows():
            datalist.append((
                row['排名'],
                row['电影名称'],
                row['链接'],
                row['导演'],
                row['评分'],
                row['评分人数'],
                row['剧情简介'],
                row['上映年份'],
                row['国家和地区'],
                row['类型']
            ))

        return render_template('index.html', movies=datalist)
    except Exception as e:
        flash(f'读取电影数据失败: {str(e)}', 'error')
        return render_template('index.html', movies=[])

# 搜索
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        keyword = request.form['movie_id'].strip()
        try:
            # 读取电影数据
            matched_movies = comments_df[comments_df['电影名称'].str.contains(keyword, case=False, na=False)]
            if matched_movies.empty:
                flash('未找到匹配的电影！', 'error')
                return render_template('search.html')

            # 从匹配的电影数据中获取第一条记录
            movie = matched_movies.iloc[0]

            movie_comments = comments_df[comments_df['电影名称'] == movie['电影名称']]['评论内容'].tolist()
            if not movie_comments:
                movie_comments = ["暂无真实评论数据"]
                flash('未找到该电影的评论', 'warning')

            # 使用分离的情感分析模块
            analysis_result = Analyzer.analyze_comments(movie_comments)

            # 准备电影数据
            movie_data = {
                '电影名称': movie['电影名称'],
                '评论内容': movie['评论内容'], 
                'sentiment': analysis_result['sentiment_counts'],  # 存储情感分析结果中的情感计数
                'comments': analysis_result['comments']  # 存储情感分析结果中的评论
            }
            return render_template('movie_detail.html', movie=movie_data)
        except Exception as e:
            flash(f'处理数据时出错: {str(e)}', 'error')
            return render_template('search.html')
    return render_template('search.html')

# 词云图
@app.route('/wordcloud')
def wordcloud():
    # 此处可以添加生成词云图逻辑
    return render_template('wordcloud.html')

# 评分分析
@app.route('/rating_analysis')
def rating_analysis():
    # 处理评分数据（按间隔分组）
    rating_bins = [8.0, 8.5, 9.0, 9.5, 10.0]
    rating_labels = ['8.0-8.5', '8.5-9.0', '9.0-9.5', '9.5-10.0']
    movies_df['评分区间'] = pd.cut(movies_df['评分'], bins=rating_bins, labels=rating_labels, right=False)
    rating_counts = movies_df['评分区间'].value_counts().to_dict()

    # 处理年份数据
    movies_df['年份'] = movies_df['上映年份'].str.extract(r'(\d{4})')[0]
    year_counts = movies_df['年份'].value_counts().sort_index().to_dict()

    # 处理评价人数数据（按百万间隔分组）
    movies_df['评分人数'] = movies_df['评分人数'].astype(int)
    eval_bins = [0, 500000, 1000000, 1500000, 2000000, 2500000, float('inf')]
    eval_labels = ['0-50万', '50-100万', '100-150万', '150-200万', '200-250万', '250万+']
    movies_df['评分人数区间'] = pd.cut(movies_df['评分人数'], bins=eval_bins, labels=eval_labels)
    eval_counts = movies_df['评分人数区间'].value_counts().to_dict()

    # 先按评价数降序排序
    df_sorted = movies_df.sort_values('评分人数', ascending=False)
    top_25_movies = df_sorted.head(25)

    movie_details = []
    for _, row in top_25_movies.iterrows():
        movie_details.append({
            '电影名称': row['电影名称'],
            '评分人数': int(row['评分人数']),
            '评分': row['评分'],
            '上映年份': row['上映年份'],
            '国家和地区': row['国家和地区']
        })

    # 将数据传递到前端
    return render_template('rating_analysis.html',  rating_data=rating_counts, year_data=year_counts, eval_data=eval_counts, movie_details=movie_details)

# 地图占比（不再需要登录）
@app.route('/area')
def area():
    try:
        # 只保留国家名称的前4个字，并去除数字
        movies_df['国家'] = movies_df['国家和地区'].str[:4].str.replace(r'\d+', '', regex=True)

        # 统计各国家的电影数量
        country_counts = movies_df['国家'].value_counts().to_dict()

        return render_template('area.html', country_data=country_counts)
    except Exception as e:
        flash(f'加载地图数据失败: {str(e)}', 'error')
        return render_template('area.html', country_data={})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
