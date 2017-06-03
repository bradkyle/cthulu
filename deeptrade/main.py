
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import numpy as np
import argparse


#git key 34f3b63339fa600452e471035e65ed7fd13ee5a0

#=======================================================================================>
# Arguments
#=======================================================================================>
parser = argparse.ArgumentParser(description='Utilizing Asynchronous Advantage Actor Critic and Meta Learning to trade')

parser.add_argument('--gamma', type=float, default=0.99, metavar='LR', help='discount rate for advantage estimation and reward discounting')

#todo a-size
#todo model path
#todo load model
#todo max episode length

parser.add_argument('--s-size', type=float, default=7056, metavar='N', help='input shape size')

#instantiate args for later usage
config = parser.parse_args()

#=======================================================================================>
# Helper Functions
#=======================================================================================>

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



#=======================================================================================>
# Actor-Critic Network
#=======================================================================================>

#todo create time step

#Actor-Critic combines the benefits of value-iteration methods
#such as Q-learning and policy-iteration methods such as Policy Gradient
#our network will estimate both a value function V(s) (how good a certain state is to be in) and a policy π(s)
#(a set of action probability outputs). These will each be separate fully-connected layers sitting at the top of
#the network. Critically, the agent uses the value estimate (the critic) to update the policy (the actor) more
#intelligently than traditional policy gradient methods.
class AC_Network():
    def __init__(self, args, scope, trainer):
        with tf.variable_scope(scope):

            self.args = args

            # Instantiates a placeholder for a tensor that will be always fed.
            # The type of elements in the tensor to be fed is float32
            self.state = tf.placeholder(shape=[None, self.args.s_size], dtype=tf.float32)
            #todo create multiple inputs representative of the asset classes

            #Adds a fully connected layer.
            # - input: flattened(more efficient) state
            # - num_outputs: 64 ?
            # - activation_fn: tf.nn.elu Computes exponential linear
            self.conv = slim.fully_connected(slim.flatten(self.state),64,activation_fn=tf.nn.elu)

            #Seen as though we are creating a Meta AC3 we will feed the previous actions/rewards to the algorithm
            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32) #
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32) #
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32) #
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, self.args.a_size, dtype=tf.float32) #

            # Adds a fully connected layer.
            # fully_connected creates a variable called weights, representing a fully connected weight matrix, which is
            # multiplied by the inputs to produce a Tensor of hidden units. If a normalizer_fn is provided
            # (such as batch_norm), it is then applied. Otherwise, if normalizer_fn is None and a biases_initializer
            # is provided then a biases variable would be created and added the hidden units. Finally, if activation_fn
            # is not None, it is applied to the hidden units as well.

            # receives the second convolutional layer after it has been flattened
            # (whilst maintaining batch size, Assumes that the first dimension represents the batch.)
            # the activation function of a node defines the output of that node given an input or set of inputs.
            hidden = tf.concat([slim.flatten(self.conv), self.prev_rewards, self.prev_actions_onehot, self.timestep], 1)

            #
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(48, state_is_tuple=True)

            ##Return a new array of given shape and type, filled with zeros.
            #use the size of the state returned by the lstm_cell initialized above
            #for both the c and h tuples
            #set the data type to NUMPY float32
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)

            # initial state = array of above initialized numpy zeros arrays
            self.state_init = [c_init, h_init]

            # add a tensorflow placeholder for the arrays with a data type of float32 and a
            # size equal to the size of the lstm_cell state initialized previousely
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])

            # create a tuple of the placeholder set initialized above
            self.state_in = (c_in, h_in)

            # Inserts a dimension of 1 into a tensor's shape.
            # Given a tensor input, this operation inserts a dimension of 1 at the dimension index axis of input's shape.
            # The dimension index axis starts at zero; if you specify a negative number for axis it is counted backward
            # from the end.
            # This operation is useful if you want to add a batch dimension to a single element. For example,
            # if you have a single image of shape [height, width, channels], you can make it a batch of 1 image with
            # expand_dims(image, 0), which will make the shape [1, height, width, channels].
            # Returns: A Tensor with the same data as input, but its shape has an additional dimension of size 1 added.
            rnn_in = tf.expand_dims(hidden, [0])

            # Returns the shape of the imageIn tensor input
            # This operation returns a 1-D integer tensor representing the shape of input.
            # Returns:A Tensor of type out_type.
            step_size = tf.shape(self.imageIn)[:1]

            # Create new instance of LSTMStateTuple
            # Instantiates the state vector of the BasicLSTM reccurrent network cell spe
            # c (c_in) Alias for field number 0
            # h (h_in) Alias for field number 0
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            # Creates a recurrent neural network specified by RNNCell cell
            # This function is functionally identical to the function rnn above, but performs fully dynamic unrolling of inputs.
            # Unlike rnn, the input inputs is not a Python list of Tensors, one for each frame. Instead, inputs may be a
            # single Tensor where the maximum time is either the first or second dimension (see the parameter time_major).
            # Alternatively, it may be a (possibly nested) tuple of Tensors, each of them having matching batch and time
            # dimensions. The corresponding output is either a single Tensor having the same number of time steps and batch
            # size, or a (possibly nested) tuple of such tensors, matching the nested structure of cell.output_size.
            #
            # cell = lstm_cell:          An instance of RNNCell created with BasicLSTMCell above
            # inputs = rnn_in:           rnn (Recurrent Neural Network) receives the hidden fully connected layer that is a direct result of the
            #                           flattening of the result of the convolutional layer that proceeded it.
            # initial_state = state_in:  An initial state for the RNN
            # sequence_length=step_size: An int32/int64 vector sized, The parameter sequence_length is optional and is used
            #                           to copy-through state and zero-out outputs when past
            #                           a batch element's sequence length. So it's more for correctness than performance, unlike in rnn().
            # time_major=False:          The shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_
            #                           size, depth]. If false, these Tensors must be shaped [batch_size, max_time, depth]. Using time_major = True
            #                           is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However,
            #                           most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major
            #                           form. * scope: VariableScope for the created subgraph; defaults to "rnn".
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in,sequence_length=step_size, time_major=False)

            # splits the lstm_state tuple back into the seperate lstm_c, lstm_h
            lstm_c, lstm_h = lstm_state

            #todo explain
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

            # Reshapes a tensor.
            # Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
            # If one component of shape is the special value -1, the size of that dimension is computed so that the total
            # size remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of shape can
            # be -1.
            # If shape is 1-D or higher, then the operation returns a tensor with shape shape filled with the values of
            # tensor. In this case, the number of elements implied by shape must be the same as the number of elements in
            # tensor.
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32) # instantiate a placeholder for the actions
            self.actions_onehot = tf.one_hot(self.actions, self.args.a_size, dtype=tf.float32) # Returns a one-hot tensor.

            # Policy output layer
            # policy π(s) (a set of action probability outputs)
            # Output layers for policy and value estimations
            # input = rnn_out:               utilizes the reshaped outputs from the dynamic_rnn above
            # num_outputs = a_size:          number of outputs is equal to the number of actions specified in the config
            # activation_fn = tf.nn.softmax: Computes softmax activations. Returns A Tensor. Has the same type as logits.
            #                               Same shape as logits. Raises: * InvalidArgumentError: if
            #                               logits is empty or dim is beyond the last dimension of logits.
            # weights_initializer = normalized_columns_initializer(0.01):
            # biases_initializer = None:
            self.policy = slim.fully_connected(rnn_out, self.args.a_size, activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)

            # Value output layer
            # value function V(s) (how good a certain state is to be in)
            # Output layers for policy and value estimations
            # input = rnn_out:               utilizes the reshaped outputs from the dynamic_rnn above
            # num_outputs = a_size:          number of outputs is equal to the number of actions specified in the config
            # activation_fn = tf.nn.softmax: Computes softmax activations. Returns A Tensor. Has the same type as logits.
            #                               Same shape as logits. Raises: * InvalidArgumentError: if
            #                               logits is empty or dim is beyond the last dimension of logits.
            # weights_initializer = normalized_columns_initializer(0.01):
            # biases_initializer = None:
            self.value = slim.fully_connected(rnn_out, 1, activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':   #todo
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 50.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))



# =======================================================================================>
# Worker Agent
# =======================================================================================>

#todo needs to be adapted to trading thus make trading first
class Worker():
    def __init__(self, name, trainer, global_episodes, args, env):
        self.args = args
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = self.args.model_path  # path to the saved model
        self.trainer = trainer  # Specifies the optimizer that is used
        self.global_episodes = global_episodes  # instance of all global episodes
        self.increment = self.global_episodes.assign_add(1)  # adds a episode to global episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        # Writes Summary protocol buffers to event files.
        # The FileWriter class provides a mechanism to create an
        # event file in a given directory and add summaries and events to it.
        # The class updates the file contents asynchronously. This allows a training program to call methods to add data
        # to the file directly from the training loop, without slowing down training.
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(self.args.s_size, self.args.a_size, self.name, trainer)

        # Copies one set of variables to another.
        # Used to set worker network parameters to those of global network.
        self.update_local_ops = update_target_graph('global', self.name)

        #capture the environment
        self.env = env

    #todo summary
    def train(self,rollout,sess,bootstrap_value):
        rollout = np.array(rollout) #
        observations = rollout[:, 0] #
        actions = rollout[:, 1] #
        rewards = rollout[:, 2] #
        timesteps = rollout[:, 3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        next_observations = rollout[:, 3] #
        values = rollout[:, 5] #

        self.pr = prev_rewards
        self.pa = prev_actions

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value]) #
        discounted_rewards = discount(self.rewards_plus, self.args.gamma)[:-1] #
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value]) #
        advantages = rewards + self.args.gamma * self.value_plus[1:] - self.value_plus[:-1] #
        advantages = discount(advantages, self.args.gamma) #

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init #set the state of the rnn to the state of the local copy of the AC Network

        #The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
        #
        #- If the key is a tf.Tensor, the value may be a Python scalar, string, list,
        #  or numpy ndarray that can be converted to the same dtype as that tensor.
        #  Additionally, if the key is a tf.placeholder, the shape of the value will be checked for compatibility with the placeholder.
        #
        #- If the key is a tf.SparseTensor, the value should be a tf.SparseTensorValue.
        #
        #- If the key is a nested tuple of Tensors or SparseTensors, the value should be a nested tuple with the same structure
        #  that maps to their corresponding values as above.

        #Each value in feed_dict must be convertible to a numpy array of the dtype of the corresponding key.
        feed_dict = {
                     self.local_AC.target_v: discounted_rewards, #
                     self.local_AC.state: np.vstack(observations), #
                     self.local_AC.prev_rewards: np.vstack(prev_rewards), #
                     self.local_AC.prev_actions: prev_actions, #
                     self.local_AC.actions: actions, #
                     self.local_AC.timestep: np.vstack(timesteps), #
                     self.local_AC.advantages: advantages, #
                     self.local_AC.state_in[0]: rnn_state[0], #
                     self.local_AC.state_in[1]: rnn_state[1] #
                    }

        #Resets resource containers on target, and close all connected sessions.
        #A resource container is distributed across all workers in the same cluster as target.
        # When a resource container on target is reset, resources associated with that container will be cleared.
        # In particular, all Variables in the container will become undefined: they lose their values and shapes.
        v_l, p_l, e_l, g_n, v_n, _ = sess.run(
                                               [ #fetches
                                               self.local_AC.value_loss, #
                                               self.local_AC.policy_loss, #
                                               self.local_AC.entropy, #
                                               self.local_AC.grad_norms, #
                                               self.local_AC.var_norms, #
                                               self.local_AC.apply_grads #
                                               ],
                                               feed_dict=feed_dict
        )

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    #todo summary
    def work(self, max_episode_length, gamma, sess, coord, saver):
        # instantiate vars
        episode_count = sess.run(self.global_episodes)
        total_steps = 0

        print("Starting worker " + str(self.number))

        # ensure that a resource is "cleaned up" when the
        # code that uses it finishes running
        with sess.as_default(), sess.graph.as_default():
            # while the tensorflow coordinator does not return should_stop()
            # execute the following code
            while not coord.should_stop():
                # utilizes the tensorflow instance and runs update on local operations
                sess.run(self.update_local_ops)

                episode_buffer = []  #
                episode_values = []  #
                episode_frames = []  #
                episode_reward = 0  # The total reward count recieved during the game episode
                episode_step_count = 0
                finished = False

                r = 0 #rewards
                a = 0 #
                t = 0

                # get the current initial state of the local Actor Critic
                rnn_state = self.local_AC.state_init

                while finished == False:

                    a_dist, v, rnn_state_new = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={
                            self.local_AC.state: [s],
                            self.local_AC.prev_rewards: [[r]],
                            self.local_AC.timestep: [[t]],
                            self.local_AC.prev_actions: [a],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]
                        }
                    )

                    # Take an action using probabilities from policy network output.
                    #todo this needs to be fixed
                    s1, s1_big, r, d, _, _ = self.env.step(a)
                    pisode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    episode_frames.append(set_image_gridworld(s1_big, reward_color, episode_reward, t))
                    total_steps += 1
                    t += 1
                    episode_step_count += 1
                    s = s1


# =======================================================================================>
# Main
# =======================================================================================>

def main():

    print()