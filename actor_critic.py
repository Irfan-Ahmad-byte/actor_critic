import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorCriticNetwork

clas Agent:
	def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
		self.gamma = gamma
		self.n_actions = n_actions
		self.action = None
		self.action_space = [i for in range(n_actions)]

		self.actor_critic = ActorCriticNetwork(n_actions  = n_actions)
		self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

	
	def choose_action(self, observatin):
		state = tf.convert_to_tensor([observation])
		_, prob = self.actor_critic(state)
		
		action_probabilities = 	tfp.distributions.Categorical(probs=prob)
		action = action_probabilities.sample()
		self.action = action

		return action.numpy()[0]

	
	def save_model(self):
		print('..............saving model............')
		self.actor_critic.save_weights(self.actor_critic.chp_file)
	
	def load_model(self):
		print('.............loading model.............')
		self.actor_critic.load_weights(self.actor_critic.chp_file)

	def learn(self, state, reward, _state, done):
		state = tf.convert_to_tensor([state], dtype=tf.float32)
		_state = tf.covnet_to_tensor([_state], dtype=tf.float32)
		reward = tf.convert_to_tensor(reward, dtype=tf.float32)

		with tf.GradientTape(persistent=True) as tape:
			state_value, prob = self.actor_critic(state)
			_state_value, _ = se;f.actor_critic(_state)
			state_value  = tf.squeeze(state_value)
			_state_value = tf.squeeze(_state_value)

			action_probs = tfp.distributions.Categorical(probs=prob)
			log_prob = action_probs.log_prob(self.action)
			
			delta = reward+self.gemma*_state_value*(1-int(done)) - state_value

			actor_loss = -log_prob*delta
			critic_loss = delta**2
			total_loss = actor_loss+critic_loss

		gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
		self.actor_critic_.optimizer.apply_gradients(zip(
					gradient, self.actor_critic.trainable_variables))
