import tensorflow.compat.v1 as tf



g = tf.Graph()
with g.as_default():   
    a = tf.constant(17.5)
    b = tf.constant(5.0)
    c = tf.add(a, b)
    print(c)
    session = tf.Session(graph=g)
    print(session.run(c))

    mathScore = [85, 99, 84, 97, 92]
    englishScore = [59, 80, 84, 68, 77]
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)
    y = (a + b) / 2
    session = tf.Session(graph=g)


print(session.run(y, feed_dict={a: mathScore, b: englishScore}))


'''
g = tf.Graph()
with g.as_default():   
	#hello=tf.constant('Hello World!')
    a = tf.Variable(500)
    b = tf.Variable(2)
    c = tf.multiply(a, b)
    init = tf.global_variables_initializer()
    session = tf.Session(graph=g)
    session.run(init)

    bb = tf.Variable(3)    
    cc = tf.multiply(b, bb)
    init = tf.global_variables_initializer()
    session.run(init)

print(session.run(c))
print(session.run(cc))
'''

'''
g = tf.Graph()
with g.as_default():   
	hello=tf.constant('Hello World!')
    
session = tf.compat.v1.Session(graph=g)
print(session.run(hello))
'''

#선형회귀
#H(x) = Wx + b
'''
xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

g = tf.Graph()
with g.as_default():   
    W = tf.Variable(tf.random.uniform([1], -100, 100))
    b = tf.Variable(tf.random.uniform([1], -100, 100))
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    H = W * X + b
    cost = tf.reduce_mean(tf.square(H - Y))
    a = tf.Variable(0.01)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)
    init = tf.global_variables_initializer()
    sess = tf.Session(graph=g)
    sess.run(init)
    for i in range(5001):
        sess.run(train, feed_dict={X: xData, Y: yData})
        if i % 500 == 0:
            print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))
            
print (sess.run(H, feed_dict={X: [8]}))
'''