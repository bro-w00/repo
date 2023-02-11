import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
	
DIVIDER = '--------------------------------------------------------------------------------'

@st.cache
def load_data():
        data = pd.read_csv('Superstore.csv')
        data2 = pd.read_excel('Super.xls',sheet_name='반품 정리')
        data2 = data2.drop_duplicates()
        df_merge = pd.merge(data, data2, how='left', on='주문 ID')
        df_merge = df_merge.drop(['행 ID','고객 이름','제품 이름'],axis=1)
        df = df_merge
        
        df.rename(columns={'주문 날짜':'OrderDate'},inplace=True)
        df.rename(columns={'배송 날짜':'ShipDate'},inplace=True)

        df['주문 ID']= df['주문 ID'].astype('str')
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df['ShipDate'] = pd.to_datetime(df['ShipDate'])
        df['배송 형태']= df['배송 형태'].astype('str')
        df['고객 ID']= df['고객 ID'].astype('str')
        df['세그먼트'] = df['세그먼트'].astype('str')
        df['도시'] = df['도시'].astype('str')
        df['시/도'] = df['시/도'].astype('str')
        df['국가/지역'] = df['국가/지역'].astype('str')
        df['지역'] = df['지역'].astype('str')
        df['제품 ID'] = df['제품 ID'].astype('str')
        df['범주'] = df['범주'].astype('str')
        df['하위 범주'] = df['하위 범주'].astype('str')
        
        df.fillna('no',inplace=True)
        df = df[df['반품']=='no']
        return df

@st.cache
def load_data_n(data):
    df = data[data['지역']=='북아시아']
    return df

@st.cache
def load_data_es(data):
    df = data[data['지역']=='동남아시아']
    return df

@st.cache
def load_data_m(data):
    df = data[data['지역']=='중앙아시아']
    return df

@st.cache
def load_data_o(data):
    df = data[data['지역']=='오세아니아']
    return df

####################################
# - Sidebar
####################################
with st.sidebar:
    st.title("비저블 지원")
    st.markdown("**- 기획서 :** 기획서 프로세스")
    st.markdown("**- 대시보드 :** 대시보드 스케치")
    
VIEW_PROPOSAL = '기획서'
VIEW_DASHBOARD = '대시보드'

sidebar = [VIEW_PROPOSAL, 
        VIEW_DASHBOARD]
add_sidebar = st.sidebar.selectbox('보고싶은 페이지를 선택하세요', sidebar)

with st.sidebar:
    add_radio = st.radio(
        "사용하고싶은 언어를 고르세요 (작동 안 함)",
        ("Korean", "English")
    )
    
data_load_state = st.text('Loading data...')
df = load_data()
df_es = load_data_es(df)
df_n = load_data_n(df)
df_m = load_data_m(df)
df_o = load_data_o(df)
data_load_state.text("반품 상품 제외한 데이터 불러오기 완료!")

a = pd.DataFrame(df_n['할인율'].describe())
a.columns = ['북아시아 할인율']

b = pd.DataFrame(df_es['할인율'].describe())
b.columns = ['동남아시아 할인율']

c = pd.DataFrame(df_m['할인율'].describe())
c.columns = ['중앙아시아 할인율']

d = pd.DataFrame(df_o['할인율'].describe())
d.columns = ['오세아니아 할인율']

e = pd.concat([a,b,c,d],axis =1)

if st.checkbox('데이터 보기'):
        st.subheader('Superstore.csv')
        st.write(df)
        
####################################
# - VIEW_PROPOSAL
####################################
if add_sidebar == VIEW_PROPOSAL:
    st.title(add_sidebar)
    st.write(DIVIDER)
    
    st.subheader('1. 문제 정의')
    st.markdown('##### 매출액은 크지만 이익은 크지 않은 동남아시아 수익 구조 개선')
    st.markdown('동남아시아 지역에서 발생하는 매출이 840,428.9613인데 비해 수익은 17,552.6913으로 현저히 적은 것을 볼 수 있다.')
    st.markdown('아래 표를 봐도, 매출의 25%를 차지하는 동남아시아가 수익에서는 4.55%밖에 되지 않는 것을 볼 수 있다.')
    st.markdown('즉, 매출액 순수익률이 10%도 되지 않을 정도로 상당히 저조하다는 것을 알 수 있다.')
    st.markdown('이에 동남아시아 지역 수익구조의 개선이 필요하다.')
    
    chart_data = pd.DataFrame(
        {
            '매출':[1055576.976, 840428.9613, 710284.014, 694706.028],
            '수익':[113672.976, 17552.6913, 123819.334, 130354.098],
            '순수익률':[113672.976/1055576.976, 17552.6913/840428.9613, 123819.334/710284.014, 130354.098/694706.028]
        },
        index = ['오세아니아','동남아시아','중앙아시아','북아시아']
    )
    
    st.dataframe(chart_data)
    
    st.write(DIVIDER)
    
    st.subheader('2. 지표 설정')
    st.markdown('##### 동남아시아 지역의 수익, 매출, 매출액 순이익률(ROS)')
    st.write('동남아시아의 수익 구조를 개선하는 것이기에 동남아시아의 수익이 중요한 지표가 될 것이다.')
    st.write('또한, 수익만 증가하는 것보다는 매출이 증가하는 것이 의미가 있을 것이기에 매출도 중요한 지표가 될 것이며, 매출 중에 수익이 차지하는 비율도 중요한 지표가 될 것이다.')
    st.write(DIVIDER)
    
    st.subheader('3. 현황 파악')
    st.markdown('##### 동남아시아 지역의 매출 및 순위')
    st.write('2위', 840428.9613)
    st.markdown('##### 동남아시아 지역의 수익 및 순위')
    st.write('4위', 17552.6913) 
    st.markdown('##### 동남아시아 지역의 매출액 순이익률(ROS) 및 순위')
    ROS = 17552.6913/840428.9613
    st.write('4위', ROS)
    st.write(DIVIDER)
    
    st.subheader('4. 평가')
    st.markdown('##### [ 평가 기준 ]')
    st.write('동남아시아 지역의 매출 2위 이상 유지')
    st.write('동남아시아 지역의 수익 2위 이상으로 상승')
    st.write('매출액 순이익률 2위 이상으로 상승')
    st.markdown('##### [ 비교 대상 ]')
    st.write('매출에서는 3,4위지만, 수익에서는 1,2위를 차지하고있는 중앙아시아, 북아시아 수준의 순이익률을 내고자한다.')
    st.write(DIVIDER)
    
    st.subheader('5. 원인 분석')
    st.markdown('##### 할인율이 높을수록 수익이 적은데, 할인율이 높은 제품을 많이 판매했다.')
    st.write('''
             동남아시아에서 매출이 잘 나오는 것은, 제품이 잘 팔리기 때문으로 분석할 수 있다.
             하지만, 수익율이 낮다는 것은 매출에 비해 수익이 떨어지는, 판매를 많이 할수록 손해가 크게 발생하는 제품을 많이 팔았기 때문으로 분석된다.
             ''')
    
    st.write('''
             아래의 두 표를 볼 경우,
             할인율이 높을수록 평균 수익이 마이너스가 되는 것을 볼 수 있는데, 할인율이 47%인 제품이 가장 많이 판매된 것을 볼 수 있다.
             ''')
    st.write('''
             또한, 다른 지역들보다 동남아시아의 평균할인율이 높은 동시에 다수의 제품에 적용됐음을 볼 수 있다.
             ''')
    a_ = pd.DataFrame(df_es.groupby('할인율')['수익'].mean()).reset_index()
    b_ = pd.DataFrame(pd.DataFrame(df_es.groupby('할인율')['수량'].sum()).reset_index()['수량'])
    a_ = pd.concat([a_,b_],axis=1)
    a_.columns = ['동남아시아 할인율', '평균 수익', '수량']
    col1,col2 = st.columns(2)
    with col1:
        st.dataframe(a_)
    with col2:
        st.dataframe(e.style.highlight_max(axis=1))
    st.write(DIVIDER)
    
    st.subheader('6. 해결 방안')
    st.markdown('##### 문제가 되는 제품의 할인율 감소 및 수익성 개선 or 같은 제품군의 비슷한 제품 홍보 증가')
    st.markdown('###### 개선하고싶은 제품 선정')
    st.write('''
             동남아시아 지역의 수익률을 개선시키기 위해서는 어떤 제품에 변화를 가져와야 할까?
             ''')
    st.write('''
             수익이 가장 낮게 계산된 거래 내역 30건을 가지고 조사한 결과, '가구' 제품 범주의 '테이블' 하위 범주에서 수익이 가장 낮은 것을 발견할 수 있었다.
             ''')
    df_tmp1 = df_es[df_es['수익']<0].sort_values('수익',ascending=True).head(30)
    df_tmp1['수익'] = df_tmp1['수익'].abs()
    df_tmp2 = df_tmp1[['범주','하위 범주','수익']]
    df_tmp2.columns=['범주','하위 범주','수익 절대값']
    ab = pd.DataFrame(df_tmp2['범주'].value_counts())
    ab.columns=['범주에 속한 수']
    bc = pd.DataFrame(df_tmp2['하위 범주'].value_counts())
    bc.columns=['하위 범주에 속한 수']
    col1,col2,col3 = st.columns(3)
    with col1:
        st.dataframe(df_tmp2)
    with col2:
        st.dataframe(ab)
    with col3:
        st.dataframe(bc)    
    st.write('''
             '테이블' 하위 범주에 속해있는 제품들의 수익률을 개선하면, 전체적인 결과에 큰 영향을 줄 수 있을 것이라 판단했다. 아래는 그에 따른 개선 방안 3가지이다.
             ''')
    st.markdown('###### 개선 방안 1 할인율 감소')
    st.write('''
             우선적으로, '가구' 제품 범주의 '테이블' 하위 범주의 할인율이 높은 것을 문제삼을 수 있다. 기본 40%가 넘는 할인율을 좀 더 낮추어 제품을 판매해도 손해가 발생하지 않도록 해야한다.
             ''')
    
    st.markdown('###### 개선 방안 2 같은 제품 군에서 수익률이 좋은 제품 홍보 증가')
    st.write('''         
             테이블 범주 내에도 판매만으로 수익이 발생하는 제품이 존재한다. 제품 ID FUR-TA-10001327은 47%의 할인율로 2개 판매 시, 18.9222의 수익이 발생한다. 따라서,
             테이블을 구매하려는 고객에게 수익률이 가장 낮은 FUR-TA-10002972를 권하기보다 FUR-TA-10001327을 더욱 권하는 전략을 취할 수 있다. 홍보 시에도 마찬가지로
             수익률이 더 좋은 제품을 더 많이 홍보하는 전략을 취해야한다.
             ''')
    st.markdown('###### 개선 방안 3 고객 니즈 판단을 통한 수익성 개선 - 낮은 가격 유지')
    st.write('''
             동남아시아에서 할인율이 높은 제품이 많이 팔렸다는 것은, 동남아시아 고객들은 가격이 저렴한 제품을 선호하는 경향이 있음을 뜻할 수 있다.
             이에 수익성을 개선시키기 위해서는 원가가 더 저렴한 제품을 가져와 판매할 필요가 있다.
             이와 더불어 해당 제품의 생산부터 유통까지의 과정 중, 비용이 많이 들어가는 부분을 찾아내어 개선시킬 필요가 있다.
             물론, 이 데이터만으로 해결할 수 없는 방법일 수 있지만, 꼭 개선되어야하는 부분이기에 적는다.
             ''')
    st.write(DIVIDER)

    st.subheader('7. 결론')
    st.markdown('##### 내가 Sanjit Engle이라면..')
    st.markdown('''
                내가 동남아시아를 담당하는 인력인 Sanjit Engle이 되었다고 생각해보았다. 매출은 잘 나오지만, 수익은 매우 떨어지는 이 상황이 만족스럽지 않았다.
                순이익률을 향상시키기 위해, 나는 수익률이 낮은 제품을 찾아보았고 해당 제품의 수익률이 낮은 이유를 데이터에서 찾아보았다. 단가가 높은 테이블은
                높은 매출을 달성하게 하였지만, 제품에 매겨진 높은 할인율은 손해를 보는 장사를 하게 하고 있었다. 이에 **_테이블 제품에서의 수익률 향상_** 방안을 모색해
                보았다.
             ''')
    st.markdown('''
                첫 번째, **_할인율을 대폭 감소시키기로 하였다._** 할인율을 감소시켜 매출은 감소하더라도 수익률을 향상시킬 수 있었다.
             ''')
    st.markdown('''
                두 번째, **_테이블을 찾는 고객에게 좀 더 수익률이 좋은 제품을 추천하는 동시에 홍보하였다._** 이를 통해, 수익률이 낮은 제품 판매량을 수익률이 높은 제품을
                판매하는 것으로 대체할 수 있었다.
             ''')
    st.markdown('''            
                마지막으로, 수익 개선만을 신경쓰다보니 매출이 떨어지는 결과를 맞이하였다. 이에 동남아시아 지역의 고객들의 니즈를 다시 한 번 생각해보았다.
                이들은 낮은 가격을 선호하는 것으로 보였다. 따라서, 전체적으로 **_원가가 낮은 제품_** 으로 스토어를 다시 구성하기로 하였다.
             ''')
    st.markdown('''
                이렇게 하여, 고객의 니즈를 충족시켜 매출을 유지시키는 동시에 수익률도 개선할 수 있었다.
                ''')
    st.write(DIVIDER)

####################################
# - VIEW_DASHBOARD
####################################
elif add_sidebar == VIEW_DASHBOARD:
    st.title(add_sidebar)
    
    fig = px.pie(df,values='매출',names='지역', title='지역 별 매출 비중', color_discrete_sequence=px.colors.sequential.RdBu, category_orders={'지역':['동남아시아','오세아니아','북아시아','중앙아시아']})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    st.write('''
             동남아시아의 지역 별 매출이 전체에서 25.5%를 차지하며, 지역 2위임을 알 수 있다.
             ''')
    
    fig = px.pie(df,values='수익',names='지역', title='지역 별 수익 비중', color_discrete_sequence=px.colors.sequential.RdBu, category_orders={'지역':['동남아시아','오세아니아','북아시아','중앙아시아']})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    
    st.markdown('##### 동남아시아 제품의 할인율에 따른 평균 수익 및 판매 수량')
    st.write('''
             동남아시아의 지역 별 수익이 전체에서 4.55%를 차지하며, 많은 차이를 보이며 지역 4위로 꼴찌임을 알 수 있다.
             ''')
    
    a = pd.DataFrame(df_es.groupby('할인율')['수익'].mean().reset_index())
    b = pd.DataFrame(df_es.groupby('할인율')['수량'].sum().reset_index())
    fig = make_subplots(specs=[[{"secondary_y": True}]], shared_xaxes=True, x_title = '할인율')
    fig.add_trace(
        go.Scatter(x=b['할인율'], y=b['수량'], name="판매 수량")
    )
    fig.add_bar(
        x=b['할인율'], y=a['수익'], name="평균 수익", marker=dict(color="MediumPurple"),secondary_y=True
    )
    fig.update_yaxes(title_text="판매 수량", secondary_y=False)
    fig.update_yaxes(title_text="평균 수익", secondary_y=True)
    st.plotly_chart(fig)
    st.write('''
             할인율이 높을수록 낮은 수익을 내게 되는데, 0.5대의 할인율에서 제품을 가장 많이 판매했음을 알 수 있다.
             ''')
    
    df_tmp1 = df_es[df_es['수익']<0].sort_values('수익',ascending=True).head(20)
    df_tmp1['수익'] = df_tmp1['수익'].abs()
    fig = px.pie(df_tmp1,values='수익',names='제품 ID', title='수익률 낮은 30개 거래 내역에 존재하는 제품 ID 비중', color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    st.write('''
             FUR-TA로 시작하는, '가구' 범주 내의 '테이블' 하위 범주의 제품이 많이 포진되어 있는 것을 볼 수 있다. 여기서 수익은 이들이 발생시킨 마이너스 수익의 절대값을 의미한다.
             ''')
    
    df_ga = df_es[df_es['범주']=='가구']
    df_ha = df_ga[df_ga['하위 범주']=='테이블']
    df_ha = df_ha.groupby('할인율')['수익'].sum().reset_index()
    st.markdown('**\'테이블\' 하위 범주 내의 할인율에 따른 수익**')
    fig = px.bar(df_ha, x='할인율', y='수익')
    st.plotly_chart(fig)
    st.markdown('테이블 제품의 수익률이 마이너스로 심각함을 알 수 있다.')
