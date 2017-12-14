from agent_dir.agent import Agent
import tensorflow as tf 
import numpy as np 
import random
from collections import deque
import sys
sys.path.append("agent_dir/")
import rank_based

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # UNKOWND ERROR for out of memory problem

tf.set_random_seed(1)

"""
Hyper Parameters:
"""
GAMMA = 0.99 # decay rate of past observations
EXPLORE = 1000000.
OBSERVE = 10000. # timesteps to observe before training
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 10000
BATCH_SIZE = 32
UPDATE_TIME = 1000

class Agent_DQN(Agent):
  def __init__(self, env, args):
    """
    Initialize every things you need here.
    For example: building your model
    """

    super(Agent_DQN,self).__init__(env)

    # if args.test_dqn:
    #   #you can load your model here
    #   print('loading trained model')

    ##################
    # YOUR CODE HERE #
    ##################
    self.prioritized = args.prioritized or False

    if self.prioritized:
      self.experience = rank_based.Experience({
        'size': REPLAY_MEMORY,
        'batch_size': BATCH_SIZE,
        'learn_start': OBSERVE,
        'total_steps': EXPLORE * 10
      })


    self.env = env
    # init replay memory
    self.replayMemory = deque()
    # init some parameters
    self.timeStep = 0
    self.epsilon = INITIAL_EPSILON
    self.actions = 4
    # init Q network
    self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

    # init Target Q Network
    self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

    # a sequence of tf operation
    self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),\
      self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2), self.W_conv3T.assign(self.W_conv3),\
      self.b_conv3T.assign(self.b_conv3), self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),\
      self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

    self.createTrainingMethod()

    # saving and loading networks
    self.saver = tf.train.Saver()
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())

    if args.test_dqn:
      #you can load your model here
      self.saver.restore(self.session, 'dqn_model')
      print('loading trained model')

  def init_game_setting(self):
    """

    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary

    """
    ##################
    # YOUR CODE HERE #
    ##################
    self.epsilon = 0.01


  def train(self):
    """
    Implement your training algorithm here
    """
    ##################
    # YOUR CODE HERE #
    ##################
    self.env.seed(11037)
    self.currentState = self.env.reset()

    episode = 0
    sum_reward = 0
    avg_sum = [] # size: 100

    # Step 3.2: run the game
    for _ in range(int(1e7)):
      action = self.make_action(self.currentState)
      nextObservation, reward, terminal, info = self.env.step(action)
      action_arr = [0.] * 4
      action_arr[action] = 1.
      action_arr = np.array(action_arr)
      self.setPerception(nextObservation, action_arr, reward, terminal)

      if reward != 0: sum_reward += reward

      if terminal:
        episode += 1
        if(len(avg_sum) > 100): del avg_sum[0]
        avg_sum.append(sum_reward)
        print('Episode: ' + str(episode), ' sum_reward: ' + '{0:.1f}'.format(sum_reward) ,' avg_reward: ' + '{0:.2f}'.format(np.mean(avg_sum)))
        sum_reward = 0
        if(episode % 30 == 0):
          if self.prioritized: path = 'prioritized_dqn_learning_record.txt'
          else: path = 'dqn_learning_record.txt'

          with open(path, 'a') as f:
            f.write('Episode: ' + str(episode) + '\tavg_reward: ' + '{0:.2f}'.format(np.mean(avg_sum)) + '\tTimeSteps: ' + str(self.timeStep) + '\tEpsilon: ' + str(self.epsilon) + '\n')
          print('-' * 80 + '\n'\
            'Episode: ' + str(episode), 'avg_reward: ' + str(np.mean(avg_sum)),\
            'TimeSteps: ' + str(self.timeStep), 'Epsilon: ' + str(self.epsilon) + \
            '\n' + '-' * 80  + '\n'
          )
        self.currentState = self.env.reset()

  def make_action(self, observation, test=True):
    """
    Return predicted action of your agent

    Input:
      observation: np.array
        stack 4 last preprocessed frames, shape: (84, 84, 4)

    Return:
      action: int
        the predicted action from trained model
    """
    ##################
    # YOUR CODE HERE #
    ##################
    # Action: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    # EX: [[ 0.03629532  0.0392677   0.03857479  0.03415339]]
    QValue = self.QValue.eval(feed_dict= { self.stateInput: [observation] })[0]
    action = 0

    if random.random() <= self.epsilon:
      action = random.randrange(self.actions)
    else:
      action = np.argmax(QValue)

    # change episilon
    if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
      self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    return action

  def setPerception(self, newState, action, reward, terminal):
    if self.prioritized:
      # tuple, like(state_t, a, r, state_t_1, t)
      self.experience.store((self.currentState, action, reward, newState, terminal))
    else:
      self.replayMemory.append((self.currentState, action, reward, newState, terminal))
      if len(self.replayMemory) > REPLAY_MEMORY: self.replayMemory.popleft()

    '''
      if TIMESTEP <= OBSERVE:
        # no training
        accumulate replay memory

      elif OBSERVE <= TIMESTEP <= OBSERVE + EXPLORE:
        # start training
        random choose the action, but trust more the prediction with timestep getting bigger

      else:
        # OBSERVE + EXPLORE <= TIMESTEP :
        START TRAINING BASE ON PREDICTION
    '''

    if self.timeStep > OBSERVE: self.trainQNetwork() # Train the network
    
    self.currentState = newState
    self.timeStep += 1

  def trainQNetwork(self):
    # Step 1: obtain random minibatch from replay memory

    if self.prioritized:
      minibatch, ISWeights, e_id = self.experience.sample(self.timeStep)
    else:
      minibatch = random.sample(self.replayMemory, BATCH_SIZE)
    state_batch = [data[0] for data in minibatch] # currentState batch
    action_batch = [data[1] for data in minibatch] # action batch
    reward_batch = [data[2] for data in minibatch] # reward batch
    nextState_batch = [data[3] for data in minibatch] # newState batch

    # Step 2: calculate y 
    y_batch = []
    # Original output format is like "[[ 0.03629532  0.0392677   0.03857479  0.03415339]]"
    QValue_batch = self.QValueT.eval(feed_dict={ self.stateInputT: nextState_batch })
    # doubleQValue_batch = self.QValue.eval(feed_dict={ self.stateInput: nextState_batch })
    for i in range(0, BATCH_SIZE):
      terminal = minibatch[i][4]

      if terminal:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

    if self.timeStep % 4 == 0:
      if self.prioritized:
        _, delta = self.session.run([self.trainStep, self.abs_errors],
          feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch,
            self.ISWeights: ISWeights
          })

        self.experience.update_priority(e_id, delta) # delta size: 32
      else:
        self.trainStep.run(feed_dict={
          self.yInput : y_batch,
          self.actionInput : action_batch,
          self.stateInput : state_batch
        })

    # save network every 100000 iteration
    if self.timeStep % 10000 == 0:
      print('# Rebalance memory tree, Network weights is saved.')
      self.experience.rebalance()
      if (self.prioritized): save_path = 'saved_dqn_prioritized/dqn'
      else: save_path = 'saved_dqn/dqn'
      self.saver.save(self.session, save_path, global_step=self.timeStep)

    if self.timeStep % UPDATE_TIME == 0:
      self.copyTargetQNetwork()

  def createTrainingMethod(self):
    self.actionInput = tf.placeholder("float", [None, self.actions])
    self.yInput = tf.placeholder("float", [None]) 
    # batch裡每一個所選的action對應的Q value加總
    Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis=1)

    if self.prioritized:
      self.ISWeights = tf.placeholder("float", [None])
      cost = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.yInput, Q_Action))
      # for prioritized delta
      self.abs_errors = tf.abs(self.yInput - Q_Action)
    else:
      # 平均cost
      cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
    self.trainStep = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99).minimize(cost)

  def createQNetwork(self):
    # network weights
    W_conv1 = self.weight_variable([8, 8, 4, 32])
    b_conv1 = self.bias_variable([32])

    W_conv2 = self.weight_variable([4, 4, 32, 64])
    b_conv2 = self.bias_variable([64])

    W_conv3 = self.weight_variable([3, 3, 64, 64])
    b_conv3 = self.bias_variable([64])

    W_fc1 = self.weight_variable([3136, 512])
    b_fc1 = self.bias_variable([512])

    W_fc2 = self.weight_variable([512,self.actions])
    b_fc2 = self.bias_variable([self.actions])

    # Input layer
    stateInput = tf.placeholder("float", [None, 84, 84, 4])

    # Hidden layers
    h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
    h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_conv3_shape = h_conv3.get_shape().as_list()
    h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
    h_fc1 = tf.nn.leaky_relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, alpha=0.01)

    # Q Value layer
    QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

    return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

  def copyTargetQNetwork(self):
    self.session.run(self.copyTargetQNetworkOperation)

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")
