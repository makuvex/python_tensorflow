#12. 파이썬 빅 데이터(Big Data) K-평균(K-means) 실습하기
'''
· Numpy: 연산 처리를 용이하게 하기 위해 사용합니다.
· Pandas: 데이터 포인트를 만들기 위해 사용합니다.
· Matplotlib: 데이터 시각화를 위해 사용합니다.
'''

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


df = pd.DataFrame(columns=['x', 'y'])

df.loc[0] = [2,3]
df.loc[1] = [2,11]
df.loc[2] = [2,18]
df.loc[3] = [4,5]
df.loc[4] = [4,7]
df.loc[5] = [5,3]
df.loc[6] = [5,15]
df.loc[7] = [6,6]
df.loc[8] = [6,8]
df.loc[9] = [6,9]
df.loc[10] = [7,2]
df.loc[11] = [7,4]
df.loc[12] = [7,5]
df.loc[13] = [7,17]
df.loc[14] = [7,18]
df.loc[15] = [8,5]
df.loc[16] = [8,4]
df.loc[17] = [9,10]
df.loc[18] = [9,11]
df.loc[19] = [9,15]
df.loc[20] = [9,19]
df.loc[21] = [10,5]
df.loc[22] = [10,8]
df.loc[23] = [10,18]
df.loc[24] = [12,6]
df.loc[25] = [13,5]
df.loc[26] = [14,11]
df.loc[27] = [15,6]
df.loc[28] = [15,18]
df.loc[29] = [18,12]

df.head(30)

sb.lmplot('x', 'y', data=df, fit_reg=False, scatter_kws={"s": 100})

plt.title('K-means Example')
plt.xlabel('x')
plt.ylabel('y')

#plt.savefig('fig1.png')
#plt.show

# 데이터 프레임을 넘파이 객체로 초기화합니다.
points = df.values

# 데이터를 기반으로 K-means 알고리즘을 수행해 클러스터 4개를 생성합니다.
kmeans = KMeans(n_clusters=6).fit(points)

# 각 클러스터들의 중심 위치를 구합니다.
print(kmeans.cluster_centers_)

# 각 데이터가 속한 클러스터를 확인합니다.
print(kmeans.labels_)

#이제 시각화를 위해서 각 데이터를 클러스터 별로 기록
df['cluster'] = kmeans.labels_
#df.head(30)
print(df.head(30))

# 클러스터링이 완료된 결과를 출력
sb.lmplot('x', 'y', data=df, fit_reg=False, scatter_kws={"s": 150}, hue="cluster")

plt.title('K-means Example')
plt.savefig('fig2.png')

