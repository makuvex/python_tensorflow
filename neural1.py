import tensorflow.compat.v1 as tf

g = tf.Graph()
with g.as_default():
    # X라는 input 값을 placeholder로 세팅, 열만 3개
    X = tf.placeholder(tf.float32, [None, 3])
    print(X)

    # X placeholder에 넣을 데이터 세팅
    input = [[1,2,3],[4,5,6],[7,8,9]]

    # 가중치 변수에 [3,2] 구조로 랜덤값을    넣는다
    W = tf.Variable(tf.random_normal([3,2]))

    # 편향변수에 [3,1] 구조로 랜덤값을 넣는다
    b = tf.Variable(tf.random_normal([3,1]))

    # Input값과 가중치를 곱하고, 편향값을 더한값을 hypothesis
    hypothesis = tf.matmul(X,W) + b

    # 텐서플로우 시작
    sess = tf.Session(graph=g)

    # 정의한 변수들 초기화
    sess.run(tf.global_variables_initializer())

    # input값 출력
    print("============== INPUT ==============")
    print(input)

    # 가충치값 출력
    print("============== W ==============")
    print(sess.run(W))

    # 편향값 출력
    print("============== b ==============")
    print(sess.run(b))

    # 최종결과
    print("============== hypothesis ==============")
    print(sess.run(hypothesis, feed_dict={X: input}))
    sess.close()
    
