import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="AI岗位薪资预测系统",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ====================== 加载模型 ======================
@st.cache_resource
def load_models():
    """加载模型和预处理器"""
    model = joblib.load('champion_salary_model_no_leak.pkl')
    scaler = joblib.load('champion_scaler_no_leak.pkl')
    feature_cols = joblib.load('feature_cols_no_leak.pkl')
    city_map = joblib.load('city_map.pkl')
    industry_map = joblib.load('industry_map.pkl')
    return model, scaler, feature_cols, city_map, industry_map


# ====================== 辅助函数 ======================
def preprocess_skills(skills_str):
    """技能预处理"""
    if pd.isna(skills_str) or skills_str == '':
        return ''
    return str(skills_str).replace('|', ' ').lower()


def extract_skills_features(skills_processed):
    """从技能文本中提取特征"""
    features = {
        'has_llm_skill': 1 if 'llm' in skills_processed or 'gpt' in skills_processed else 0,
        'has_rag': 1 if 'rag' in skills_processed or 'vector' in skills_processed else 0,
        'has_finetune': 1 if 'fine-tun' in skills_processed or 'fine tuning' in skills_processed else 0,
        'has_prompt': 1 if 'prompt' in skills_processed else 0,
        'has_kubernetes': 1 if 'kubernetes' in skills_processed or 'k8s' in skills_processed else 0,
        'has_mlops': 1 if 'mlops' in skills_processed or 'mlflow' in skills_processed else 0,
        'has_cloud': 1 if 'cloud' in skills_processed or 'aws' in skills_processed or 'azure' in skills_processed or 'gcp' in skills_processed else 0,
        'has_python': 1 if 'python' in skills_processed else 0,
        'has_dl': 1 if 'pytorch' in skills_processed or 'tensorflow' in skills_processed or 'cuda' in skills_processed else 0,
        'has_data_eng': 1 if 'spark' in skills_processed or 'hadoop' in skills_processed or 'airflow' in skills_processed or 'dbt' in skills_processed else 0,
    }

    # 组合特征
    features['llm_kubernetes_combo'] = 1 if (features['has_llm_skill'] and features['has_kubernetes']) else 0
    features['ai_mlops_combo'] = 1 if (features['has_llm_skill'] and features['has_mlops']) else 0
    features['full_stack_ai'] = 1 if (
                features['has_llm_skill'] and features['has_cloud'] and features['has_python']) else 0

    return features


def get_city_encoding(city, city_map):
    """获取城市编码"""
    return city_map.get(city, 150000)


def get_industry_encoding(industry, industry_map):
    """获取行业编码"""
    return industry_map.get(industry, 150000)


def format_currency(amount):
    """格式化货币"""
    return f"${amount:,.0f}"


def get_salary_grade(salary):
    """获取薪资等级"""
    if salary < 100000:
        return "Entry Level", "📘"
    elif salary < 150000:
        return "Mid Level", "📙"
    elif salary < 200000:
        return "Upper-Mid Level", "📗"
    elif salary < 300000:
        return "Senior Level", "📕"
    else:
        return "Elite Level", "👑"


def predict_salary(job_info, model, scaler, feature_cols, city_map, industry_map):
    """预测薪资"""
    # 预处理技能
    skills_processed = preprocess_skills(job_info.get('required_skills', ''))
    skills_feat = extract_skills_features(skills_processed)

    # 构建特征向量
    features = {
        'years_of_experience': job_info.get('years_of_experience', 3),
        'exp_level_numeric': job_info.get('exp_level_numeric', 2),
        'skill_count': len(str(job_info.get('required_skills', '')).split('|')) if job_info.get(
            'required_skills') else 0,
        'company_size_encoded': job_info.get('company_size_encoded', 3),
        'is_fully_remote': 1 if job_info.get('remote_work') == 'Fully Remote' else 0,
        'is_hybrid': 1 if job_info.get('remote_work') == 'Hybrid' else 0,
        'is_senior_role': job_info.get('is_senior', 0),
        **skills_feat,
        'demand_score': job_info.get('demand_score', 75),
        'demand_growth_yoy_pct': job_info.get('demand_growth', 15),
        'ai_salary_premium_pct': job_info.get('ai_premium', 10),
        'benefits_score_10': job_info.get('benefits_score', 7),
        'city_encoded': get_city_encoding(job_info.get('city', 'San Francisco'), city_map),
        'industry_encoded': get_industry_encoding(job_info.get('industry', 'Technology'), industry_map)
    }

    # 转换为DataFrame并预测
    X = pd.DataFrame([features])[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    salary = model.predict(X_scaled)[0]

    return salary, features, skills_feat


# ====================== 主页面 ======================
def main():
    # 加载模型
    try:
        model, scaler, feature_cols, city_map, industry_map = load_models()
        st.success("✅ 模型加载成功！")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.info("请确保已运行训练脚本生成模型文件")
        return

    # 标题
    st.title("💰 AI岗位薪资预测系统")

    # 侧边栏 - 输入参数
    with st.sidebar:
        st.header("📝 岗位信息配置")
        st.markdown("---")

        # 基本信息
        st.subheader("👤 基本信息")
        city = st.selectbox(
            "城市",
            options=['San Francisco', 'New York', 'Seattle', 'Boston', 'Los Angeles',
                     'Austin', 'Chicago', 'London', 'Singapore', 'Tokyo', 'Beijing',
                     'Bangalore', 'Paris', 'Berlin', 'Sydney', 'Toronto', 'Dubai', 'Zurich'],
            index=0
        )

        industry = st.selectbox(
            "行业",
            options=['Technology', 'Finance', 'Healthcare', 'Automotive', 'Consulting',
                     'Manufacturing', 'Energy', 'Retail', 'Media', 'Government', 'Education', 'Research'],
            index=0
        )

        years_exp = st.slider("工作经验 (年)", 0, 20, 5, 1)

        exp_level = st.select_slider(
            "职级",
            options=['Entry (0-2 yrs)', 'Mid (3-5 yrs)', 'Senior (6-9 yrs)', 'Lead (10+ yrs)'],
            value='Senior (6-9 yrs)'
        )
        exp_level_map = {'Entry (0-2 yrs)': 1, 'Mid (3-5 yrs)': 2, 'Senior (6-9 yrs)': 3, 'Lead (10+ yrs)': 4}
        exp_level_numeric = exp_level_map[exp_level]

        company_size = st.select_slider(
            "公司规模",
            options=['Startup (1-50)', 'SME (51-500)', 'Mid-size (501-5000)', 'Enterprise (5000+)',
                     'Big Tech (FAANG+)'],
            value='Big Tech (FAANG+)'
        )
        company_size_map = {
            'Startup (1-50)': 1, 'SME (51-500)': 2,
            'Mid-size (501-5000)': 3, 'Enterprise (5000+)': 4, 'Big Tech (FAANG+)': 5
        }
        company_size_encoded = company_size_map[company_size]

        remote_work = st.radio(
            "工作模式",
            options=['On-site', 'Hybrid', 'Fully Remote'],
            index=2
        )

        st.markdown("---")

        # 技能信息
        st.subheader("🔧 技能信息")

        # 预设技能模板
        skill_templates = {
            "选择模板...": "",
            "普通后端工程师": "Python|SQL|Git|Docker",
            "Prompt工程师": "Python|LLM APIs|Prompt Design|Documentation|NLP",
            "LLM工程师": "Python|LLM|Fine-tuning|Vector DBs|RAG",
            "MLOps工程师": "Python|Kubernetes|Docker|MLflow|CI/CD|Cloud",
            "全栈AI工程师": "Python|LLM|Kubernetes|Cloud|PyTorch|RAG|MLOps",
            "RAG专家": "Python|LLM|Vector DBs|LangChain|Embeddings|Search Systems",
            "计算机视觉工程师": "Python|PyTorch|TensorFlow|CUDA|OpenCV|CNNs",
            "数据科学家": "Python|SQL|Statistics|ML Algorithms|Scikit-learn"
        }

        selected_template = st.selectbox("技能模板", options=list(skill_templates.keys()))

        if selected_template != "选择模板...":
            default_skills = skill_templates[selected_template]
        else:
            default_skills = "Python|PyTorch|LLM|Cloud"

        skills = st.text_area(
            "技能列表 (用 | 分隔)",
            value=default_skills,
            help="例如: Python|PyTorch|LLM|Kubernetes|Cloud"
        )

        st.markdown("---")

        # 市场因素
        st.subheader("📊 市场因素")
        demand_score = st.slider("市场需求分数", 0, 100, 85, 5, help="职位市场需求热度")
        demand_growth = st.slider("需求增长率 (%)", -20, 100, 25, 5, help="年对年需求增长")
        ai_premium = st.slider("AI技能溢价 (%)", 0, 30, 15, 2, help="AI技能带来的薪资溢价")
        benefits_score = st.slider("福利评分", 1, 10, 8, 1, help="公司福利综合评分")

        # 是否为高级职位
        is_senior = 1 if exp_level_numeric >= 3 else 0

    # 主内容区域
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 薪资预测结果")

        # 构建输入
        job_info = {
            'years_of_experience': years_exp,
            'exp_level_numeric': exp_level_numeric,
            'company_size_encoded': company_size_encoded,
            'remote_work': remote_work,
            'is_senior': is_senior,
            'city': city,
            'industry': industry,
            'required_skills': skills,
            'demand_score': demand_score,
            'demand_growth': demand_growth,
            'ai_premium': ai_premium,
            'benefits_score': benefits_score
        }

        # 预测
        salary, features, skills_feat = predict_salary(job_info, model, scaler, feature_cols, city_map, industry_map)
        salary_grade, grade_icon = get_salary_grade(salary)

        # 显示结果
        col1_1, col1_2 = st.columns([2, 1])
        with col1_1:
            st.metric("💰 预计年薪", format_currency(salary))


        with col1_2:
            st.metric("📊 薪资等级", f"{grade_icon} {salary_grade}")

        # 薪资仪表盘
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=salary,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "年薪 (USD)"},
            gauge={
                'axis': {'range': [None, 400000], 'tickformat': '$,.0f'},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 100000], 'color': "#ffcccc"},
                    {'range': [100000, 150000], 'color': "#ffffcc"},
                    {'range': [150000, 200000], 'color': "#ccffcc"},
                    {'range': [200000, 300000], 'color': "#ccffff"},
                    {'range': [300000, 400000], 'color': "#d4f1f9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': salary
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # 薪资对比
        st.subheader("📈 薪资对比")

        # 同城市平均薪资估算
        city_avg = city_map.get(city, 150000)
        industry_avg = industry_map.get(industry, 150000)

        comp_data = pd.DataFrame({
            '类别': ['你的预测', f'{city}平均', f'{industry}行业平均', '全国平均'],
            '薪资': [salary, city_avg, industry_avg, 150000]
        })

        fig = px.bar(comp_data, x='类别', y='薪资', title='薪资对比',
                     color='薪资', color_continuous_scale='viridis',
                     text='薪资')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("🔍 技能分析")

        # 技能雷达图
        skill_categories = {
            'LLM/AI': skills_feat['has_llm_skill'],
            'RAG': skills_feat['has_rag'],
            '微调': skills_feat['has_finetune'],
            'Prompt': skills_feat['has_prompt'],
            'MLOps': skills_feat['has_mlops'],
            'K8s': skills_feat['has_kubernetes'],
            '云平台': skills_feat['has_cloud'],
            'Python': skills_feat['has_python'],
            '深度学习': skills_feat['has_dl'],
            '数据工程': skills_feat['has_data_eng']
        }

        # 计算技能得分
        skill_score = sum(skill_categories.values()) / len(skill_categories) * 100

        st.metric("🏆 综合技能得分", f"{skill_score:.0f}/100")

        # 展示技能详情
        st.subheader("📋 技能清单")

        skill_status = []
        for skill, has in skill_categories.items():
            status = "✅" if has else "❌"
            skill_status.append(f"{status} {skill}")

        # 分两列显示
        col_a, col_b = st.columns(2)
        for i, skill in enumerate(skill_status):
            if i % 2 == 0:
                col_a.write(skill)
            else:
                col_b.write(skill)

        # 技能溢价分析
        st.subheader("💰 技能溢价分析")

        premium_data = []
        if skills_feat['has_finetune']:
            premium_data.append(("微调专家", "+10.8%"))
        if skills_feat['has_rag']:
            premium_data.append(("RAG专家", "+3.7%"))
        if skills_feat['has_mlops']:
            premium_data.append(("MLOps", "+1.3%"))
        if skills_feat['has_llm_skill'] and not (skills_feat['has_finetune'] or skills_feat['has_rag']):
            premium_data.append(("基础LLM", "-1.4%"))

        if premium_data:
            for skill, premium in premium_data:
                if "+" in premium:
                    st.success(f"🎯 {skill}: {premium} 薪资溢价")
                else:
                    st.warning(f"⚠️ {skill}: {premium} 单独LLM需要配合其他技能")
        else:
            st.info("💡 建议学习高价值技能: 微调、RAG、MLOps")

        # 技能提升建议
        st.subheader("💡 技能提升建议")

        missing_skills = []
        if not skills_feat['has_finetune']:
            missing_skills.append("微调 (Fine-tuning) - 最高溢价技能")
        if not skills_feat['has_rag']:
            missing_skills.append("RAG - 高需求技能")
        if not skills_feat['has_mlops']:
            missing_skills.append("MLOps - 工程化能力")
        if not skills_feat['has_cloud']:
            missing_skills.append("云平台 - 基础能力")

        if missing_skills:
            for skill in missing_skills[:3]:
                st.info(f"📚 建议学习: {skill}")
        else:
            st.success("🎉 技能组合优秀！你已经掌握了高价值技能")

    # 第三行 - 市场分析和职业建议
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.header("🌍 市场洞察")

        # 需求趋势
        demand_trend = pd.DataFrame({
            '年份': [2024, 2025, 2026],
            'AI岗位需求指数': [60, 100, 140]
        })
        fig = px.line(demand_trend, x='年份', y='AI岗位需求指数',
                      title='AI岗位需求趋势预测',
                      markers=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # 城市薪资排名
        top_cities = sorted(city_map.items(), key=lambda x: x[1], reverse=True)[:10]
        city_df = pd.DataFrame(top_cities, columns=['城市', '平均薪资'])

        fig = px.bar(city_df, x='城市', y='平均薪资', title='全球AI岗位薪资排名',
                     color='平均薪资', color_continuous_scale='reds',
                     text='平均薪资')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.header("🚀 职业发展建议")

        # 根据当前经验水平给出建议
        st.subheader("📌 下一步发展路径")

        if years_exp < 3:
            st.write("**当前阶段: 初级工程师**")
            st.write("🎯 学习重点:")
            st.write("  - 掌握Python和基础数据结构")
            st.write("  - 学习PyTorch/TensorFlow")
            st.write("  - 积累项目经验")
            st.write("📈 预计2年后薪资: +30-50%")
        elif years_exp < 6:
            st.write("**当前阶段: 中级工程师**")
            st.write("🎯 学习重点:")
            st.write("  - 深入学习LLM应用")
            st.write("  - 掌握RAG和微调技术")
            st.write("  - 学习MLOps和云平台")
            st.write("📈 预计3年后薪资: +40-60%")
        else:
            st.write("**当前阶段: 高级/专家**")
            st.write("🎯 发展方向:")
            st.write("  - 系统架构设计")
            st.write("  - 技术团队管理")
            st.write("  - 跨领域技术整合")
            st.write("📈 预计薪资上限: $300k+")

        st.markdown("---")

        # 热门技能趋势
        hot_skills = {
            "LLM应用": 95,
            "RAG": 90,
            "微调": 88,
            "MLOps": 85,
            "K8s": 80,
            "云原生": 82
        }

        fig = px.bar(x=list(hot_skills.values()), y=list(hot_skills.keys()),
                     orientation='h', title='2026年热门技能需求度',
                     color=list(hot_skills.values()), color_continuous_scale='viridis')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # 底部信息
    st.markdown("---")
    st.caption("💡 提示: 本预测基于2025-2026年AI岗位数据训练，模型R²=0.80，仅供参考")

    # 导出结果
    if st.button("📄 导出预测报告"):
        skills_list = skills.replace('|', ', ')
        report_lines = [
            "========================================",
            "AI岗位薪资预测报告",
            "========================================\n",
            "岗位信息:",
            f"- 城市: {city}",
            f"- 行业: {industry}",
            f"- 工作经验: {years_exp}年",
            f"- 职级: {exp_level}",
            f"- 公司规模: {company_size}",
            f"- 工作模式: {remote_work}\n",
            "技能清单:",
            f"- {skills_list}\n",
            "预测结果:",
            f"- 预计年薪: {format_currency(salary)}",
            f"- 薪资等级: {salary_grade}\n",
            "技能分析:",
            f"- 综合技能得分: {skill_score:.0f}/100",
        ]

        # 添加高价值技能
        high_value_skills = [s for s, v in skill_categories.items() if v]
        if high_value_skills:
            report_lines.append(f"- 高价值技能: {', '.join(high_value_skills)}")

        report_lines.append("\n========================================")
        report = "\n".join(report_lines)
        st.download_button("下载报告", report, file_name="salary_report.txt")


if __name__ == "__main__":
    main()