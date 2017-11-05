import npyreader_v1
import numpy as np
import tensorflow as tf
import pickle

n_hidden_1 = 128
n_hidden_2 = 128
n_input = 35 * 35
n_input = 96
n_output = 7
batch_size = 10

learning_rate = 0.001
training_epochs = 15

display_step = 1

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

datpath = "./Chords_m1.txt"
npypath = "./npystack_m1.npy"
reader = npyreader_v1.Reader(datpath, npypath)

test_x = np.load("./result/npy/001_G.wav.npy")
test_x = test_x.reshape(1, 96)
test_y = np.array([[0, 0, 0, 0, 0, 0, 1]])

temp_weights = None
temp_bias = None

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 20
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = reader.get_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Try:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("==== End ====")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test case:", accuracy.eval({x: test_x, y: test_y}))
    
    temp_weights = {
        'h1': weights['h1'].eval(),
        'h2': weights['h2'].eval(),
        'out': weights['out'].eval()
    }
    temp_bias = {
        'b1': biases['b1'].eval(),
        'b2': biases['b2'].eval(),
        'out': biases['out'].eval()
    }

f = open('weights.txt', 'wb')
pickle.dump(temp_weights, f)
f.close()

f = open('biases.txt', 'wb')
pickle.dump(temp_bias, f)
f.close()
