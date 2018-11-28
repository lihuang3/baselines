import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc, flatten_two_dims, unflatten_first_dim
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd

from baselines.common.models import get_network_builder

import gym, numpy as np


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, play = False, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        if play:
            a, v, state, neglogp = self._evaluate([self.pd.mode(), self.vf, self.state, self.neglogp], observation, **extra_feed)
        else:
            a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)

        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent, recurrent_tensors = policy_network(encoded_x)

            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                vf_latent, _ = _v_net(encoded_x)

        policy = PolicyWithValueBeta(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms



class PolicyWithValueBeta(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)


        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:, 0]

        with tf.variable_scope("feature_extractor"):
            self.last_observ = tf.placeholder(dtype=tf.int32,
                    shape=(1, None) + env.observation_space.shape, name='last_ob')

            self.observs = tf.placeholder(dtype=tf.int32,
                      shape=(None, None) + env.observation_space.shape, name='mb_obs')

            self.acs = self.pdtype.sample_placeholder([None, None], name='acs')
            self.features = self.get_features(env, self.observs, reuse=False)
            self.last_features = self.get_features(env, self.last_observ, reuse=True)
            self.next_features = tf.concat([self.features[1:,:], self.last_features], 0)


        self.loss = self.get_resloss(env)


    def eval_loss(self, ob, last_ob, acs):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.observs: ob, self.last_observ: last_ob, self.acs: acs}
        return sess.run(self.loss, feed_dict)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, play=False, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        if play:
            a, v, state, neglogp = self._evaluate([self.pd.mode(), self.vf, self.state, self.neglogp], observation,
                                                  **extra_feed)
        else:
            a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation,
                                                  **extra_feed)

        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def get_loss(self, env):

        acs = tf.one_hot(self.acs, env.action_space.n, axis=2)
        sh = tf.shape(acs)
        acs = flatten_two_dims(acs)

        def add_ac(x):
            return tf.concat([x, acs], axis=-1)
        scope ="feature_extractor" + "_loss"
        with tf.variable_scope(scope):
            hidsize = 256
            activ = tf.nn.relu
            x = flatten_two_dims(self.features)
            x = activ(fc(add_ac(x), scope=scope+"_fc1", nh = hidsize))
            x = activ(fc(x, scope = scope+"_fc2", nh = hidsize))

            n_out_features = self.next_features.get_shape()[-1].value
            x = fc(x, scope= scope+"_fc3", nh=n_out_features)
            x = unflatten_first_dim(x, sh)

        return tf.reduce_mean((x - tf.stop_gradient(self.next_features)) ** 2, -1)

    def get_resloss(self, env):
    
        acs = tf.one_hot(self.acs, env.action_space.n, axis=2)
        sh = tf.shape(acs)
        acs = flatten_two_dims(acs)
    
        def add_ac(x):
            return tf.concat([x, acs], axis=-1)
    

        with tf.variable_scope("feature_extractor_resloss"):
<<<<<<< HEAD
            hidsize = 128
=======
            hidsize = 256
>>>>>>> origin/master
            x = flatten_two_dims(self.features)
            x = tf.layers.dense(add_ac(x), hidsize, activation=tf.nn.leaky_relu)
    
            def residual(x):
                res = tf.layers.dense(add_ac(x), hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), hidsize, activation=None)
                return x + res
    
<<<<<<< HEAD
            for _ in range(2):
=======
            for _ in range(4):
>>>>>>> origin/master
                x = residual(x)
            n_out_features = self.next_features.get_shape()[-1].value
            x = tf.layers.dense(add_ac(x), n_out_features, activation=None)
            x = unflatten_first_dim(x, sh)
        return tf.reduce_mean((x - tf.stop_gradient(self.next_features)) ** 2, -1)
    
    def get_invloss(self, env):
        activ = tf.nn.relu
        hidsize = 128
        with tf.variable_scope("feature_extractor"):
            x = tf.concat([self.features, self.next_features], 2)
            sh = tf.shape(x)
            x = flatten_two_dims(x)
            x = activ(fc(x, nh=hidsize))
            x = fc(x, nh=env.action_space.n)
            param = unflatten_first_dim(x, sh)
            idfpd = self.pdtype.pdfromflat(param)

        return idfpd.neglogp(self.acs)

    def get_features(self, env, obs, reuse = False, normalize_observations = False, **extra_feed):
        """
        Compute forward and inverse loss, given the observation(s), action(s)
        next_observation(s)

        Parameters:
        ----------

        x         observation data (either single or a batch)
        reuse       True or False
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        cnn features
        """
        network_type = 'custom_cnn'
        ob_space = env.observation_space
        X = obs

        x_has_timesteps = (X.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(X)
            X = flatten_two_dims(X)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope("feature_extractor", reuse=reuse):
            feature_network = get_network_builder(network_type)(**{})
            feature_tensor, recurrent_tensors = feature_network(encoded_x)

        if x_has_timesteps:
            feature_tensor = unflatten_first_dim(feature_tensor, sh)
        return feature_tensor

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)
