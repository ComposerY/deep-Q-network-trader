from datetime import datetime
import random

from matplotlib import pyplot as plt

from load_data import fetch_data, plot_prices
import numpy as np
import tensorflow as tf

class QLearningDecisionPolicy:

    def __init__(self, actions, input_dim):
        self.epsilon = 0.95
        self.gamma = 0.5

        self.actions = actions
        output_dim = len(actions)

        h1_dim = 200
        h2_dim = 100
        h3_dim = 50

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])

        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        W2 = tf.Variable(tf.random_normal([h1_dim, h2_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[h2_dim]))
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        W3 = tf.Variable(tf.random_normal([h2_dim, h3_dim]))
        b3 = tf.Variable(tf.constant(0.1, shape=[h3_dim]))
        h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

        W4 = tf.Variable(tf.random_normal([h3_dim, output_dim]))
        b4 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h3, W4) + b4)

        loss = tf.square(self.y - self.q)

        self.train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def select_action(self, current_state, step):
        threshold = min(self.epsilon, step / 1000.)
        if random.random() < threshold:
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]

        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]

        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        current_action_idx = self.actions.index(action)
        action_q_vals[0, current_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


def run_simulation(policy, initial_budget, initial_num_products, prices, hist, fee=0.003, learn=True):
    budget = initial_budget
    num_products = initial_num_products
    product_value = 0

    transitions = list()

    for i in range(len(prices) - hist - 1):
        if i % 1000 == 0:
            print('progress {:.2f}%'.format(float(100 * i) / (len(prices) - hist - 1)))

        current_state = np.asmatrix(np.hstack((prices[i:i + hist], budget, num_products)))

        current_portfolio = budget + num_products * product_value

        action = policy.select_action(current_state, i)

        product_value = float(prices[i + hist])
        if action == 'Buy' and budget >= product_value:
            budget -= (product_value * (1. + fee))
            num_products += 1
        elif action == 'Sell' and num_products > 0:
            budget += (product_value * (1. - fee))
            num_products -= 1
        else:
            action = 'Hold'

        new_portfolio = budget + num_products * product_value

        if learn:
            reward = new_portfolio - current_portfolio

            next_state = np.asmatrix(np.hstack((prices[i + 1:i + hist + 1], budget, num_products)))
            transitions.append((current_state, action, reward, next_state))

            policy.update_q(current_state, action, reward, next_state)

    portfolio = budget + num_products * product_value

    return portfolio


def run_simulations(policy, budget, num_products, prices, hist, fee=0.003):
    num_tries = 10
    final_portfolios = list()
    for _ in range(num_tries):
        portfolio = run_simulation(policy, budget, num_products, prices, hist, fee)
        final_portfolios.append(portfolio)
        print('Final portfolio: ${}'.format(portfolio))
    plt.title('Final Portfolio Value')
    plt.xlabel('Simulation #')
    plt.ylabel('Net worth')
    plt.plot(final_portfolios)
    plt.show()


if __name__ == "__main__":
    prices = fetch_data("ETH-USD", 3, datetime(2016, 6, 1), datetime(2018, 1, 25), 1, "crypto_prices_1min.npy")

    n = len(prices)
    n_train = int(n * 0.90)
    train_prices = prices[:n_train]
    test_prices = prices[n_train:]
    print(train_prices)
    print(test_prices)

    plot_prices(train_prices, test_prices)

    actions = ['Buy', 'Sell', 'Hold']
    hist = 400
    policy = QLearningDecisionPolicy(actions, hist + 2)
    budget = 100000.0
    num_products = 0
    run_simulations(policy, budget, num_products, train_prices, hist)

    portfolio = run_simulation(policy, budget, num_products, test_prices, hist, learn=False)
    print(portfolio)
