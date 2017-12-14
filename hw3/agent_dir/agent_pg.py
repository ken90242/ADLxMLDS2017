from agent_dir.agent import Agent
import numpy as np
import pickle

class Agent_PG(Agent):
  def __init__(self, env, args):
    """
    Initialize every things you need here.
    For example: building your model
    """

    super(Agent_PG,self).__init__(env)

    if args.test_pg:
      #you can load your model here
      self.model = pickle.load(open('pong_model.h5', 'rb'))
      print('loading trained model')

    ##################
    # YOUR CODE HERE #
    ##################


  def init_game_setting(self):
    """

    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary

    """
    ##################
    # YOUR CODE HERE #
    self.prev_x = None
    ##################
    pass


  def train(self):
    """
    Implement your training algorithm here
    """
    ##################
    # YOUR CODE HERE #
    ##################
    pass


  def make_action(self, observation, test=True):
    """
    Return predicted action of your agent

    Input:
      observation: np.array
        current RGB screen of game, shape: (210, 160, 3)

    Return:
      action: int
        the predicted action from trained model
    """
    ##################
    # YOUR CODE HERE #
    cur_x = self.prepro(observation)
    x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(80 * 80)
    self.prev_x = cur_x

    # forward the policy network and sample an action from the returnrned probability
    aprob, h = self.policy_forward(x)

    # Actions: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    # print(action)
    ##################
    return action
    # return self.env.get_random_action()

  def prepro(self, I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop，只留中間的畫素
    I = I[::2, ::2, 0] # downsample by factor of 2，每次跳兩步選取
    I[I == 144] = 0 # erase background (background type 1)，把所有為144顏色的給幹掉
    I[I == 109] = 0 # erase background (background type 2)，把所有為109顏色的給幹掉
    # print(I)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1 # 因為球跟球板不是那些顏色，可以安心保留為1
    # print(I)
    return I.astype(np.float).ravel()  # flatten the martrix to vector

  def policy_forward(self, x):
    h = np.dot(self.model['W1'], x)
    # 在numpy裡面，可以這樣用，h矩陣裡<0都變成0
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(self.model['W2'], h) # dot不只可以用在矩陣相乘，向量相乘也可以用
    p = self.sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

  def sigmoid(self, x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

