# coding=utf-8
import os
import numpy as np


def gen_npz(demo_folder, npz_folder, use_visual_obs):

    print("=" * 5 + " Generating demo npz " + "=" * 5, end='\n\n')

    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)

    obs_name = 'visual_obs' if use_visual_obs else 'vector_obs'
    epi_count = 0
    demo_count = 0
    for f in os.listdir(demo_folder):
        path = os.path.join(demo_folder, f)
        if os.path.isfile(path) and f[-5:] == '.demo':
            # get ndarray from .demo
            # a list of episodes
            print('-' * 20)
            print("Demo:", f)
            obs, actions, rewards, done = _demo_to_np(
                path,
                use_visual_obs=use_visual_obs,
                action_dtype=np.float32,
                done_mode=False)
            length = len(rewards)
            epi_count += length

            demo_count += 1
            for n in range(length):  # Episodes
                print("Epi: {}, {}: {}, actions: {}, rewards: {}, done: {}".
                      format(n, obs_name, obs[n].shape, actions[n].shape,
                             rewards[n].shape, done[n].shape))

                npz_name = os.path.join(npz_folder, '{}_{}'.format(f, n))
                epi_data = {
                    obs_name: np.array(obs[n]),  # [0, 1], float32
                    'actions': np.array(
                        actions[n]
                    ),  # action branching (no flatten), uint8 or float32
                    'rewards': np.array(rewards[n]),  # float32
                    'done': np.array(done[n])  # boolean
                }
                np.savez(npz_name, **epi_data)

            # Check mean reward
            _r = 0
            for n in range(length):
                for r in rewards[n]:
                    _r += r
            _r /= length
            print('Mean reward:', _r)
            print('-' * 20)

    print('\nGenerate {} episodes from {} demo file'.format(epi_count, demo_count))
    print("=" * 5 + ' Done ' + "=" * 5)


def load_npz(npz_folder,
             use_visual_obs,
             visual_uint8=True,
             nd_array=True,
             make_video=False):
    if not use_visual_obs and make_video:
        raise ValueError("Can not make video from vector obs")

    obs = []
    actions = []
    rewards = []
    done = []
    gif_folder = os.path.join(npz_folder, 'demo_video')
    if not os.path.exists(gif_folder):
        os.makedirs(gif_folder)

    print("=" * 5 + ' Load npz ' + "=" * 5)
    for f in os.listdir(npz_folder):
        path = os.path.join(npz_folder, f)
        if os.path.isfile(path):
            print("npz:", f)
            _obs, a, r, d = _load_npz_demo(
                path, use_visual_obs)  # Return one episode
            # turn to uint8
            if use_visual_obs and visual_uint8:
                _obs = np.clip(np.around(np.array(_obs * 255)), 0,
                               255).astype(np.uint8)

            obs.extend(_obs)
            actions.extend(a)
            rewards.extend(r)
            done.extend(d)
            print('-' * 20)

            # gif test
            if make_video:
                import utility
                img_list = [img for img in obs]
                utility.make_video(img_list,
                                   os.path.join(gif_folder, f + ".mp4"))

    if nd_array:
        obs = np.array(obs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        done = np.array(done)

    print("=" * 5 + ' Done ({}) '.format(len(obs)) + "=" * 5)

    return obs, actions, rewards, done


def _demo_to_np(demo_path,
                use_visual_obs,
                obs_dtype=np.float32,
                action_dtype=np.uint8,
                reward_dtype=np.float32,
                done_dtype=np.bool,
                done_mode=True):
    """Convert .demo file to numpy array

    First 32 bytes of file dedicated to meta-data (In demo_loader.py).
    Thus, the length of ndarray will one less than .demo.

    :param done_mode: True, .demo 中的每段 trajectory，只有最後有結束的 (done[i] = True)，才會記錄進去 (one episode)

    Return:
        A list of ndarray
    E.g.
    data['visual_obs'] = (n, [len, H, W, C])
    n: number of episodes
    """

    # e.g. demo_buffer
    # 'done' : (2806,),
    # 'rewards' : (2806,).
    # 'visual_obs0' : (2806, 72, 128, 3), # visual_obs: [0, 1]
    # 'actions' : (2806, 3)
    from mlagents.trainers.demo_loader import load_demonstration, make_demo_buffer, demo_to_buffer
    brain_parameters, demo_buffer = demo_to_buffer(demo_path, 1)
    # print(str(brain_parameters))
    print('Demo[', str(demo_buffer.update_buffer), ']', end='\n\n')
    update_buffer = demo_buffer.update_buffer
    obs, actions, rewards, done = [], [], [], []
    obs_name = 'visual_obs0' if use_visual_obs else 'vector_obs'
    length = len(update_buffer['done'])
    ids = [i for i in range(length) if update_buffer['done'][i]]
    if len(ids) == 0:
        ids.append(length - 1)

    for i in range(len(ids)):
        s = (ids[i - 1] + 1) if i != 0 else 0
        e = ids[i] + 1
        obs.append(np.array(update_buffer[obs_name][s:e], dtype=obs_dtype))
        actions.append(
            np.array(update_buffer['actions'][s:e], dtype=action_dtype))
        rewards.append(
            np.array(update_buffer['rewards'][s:e], dtype=reward_dtype))
        done.append(np.array(update_buffer['done'][s:e], dtype=done_dtype))

    if not done_mode and ids[-1] != length - 1:

        obs.append(
            np.array(update_buffer[obs_name][ids[-1] + 1:], dtype=obs_dtype))
        actions.append(
            np.array(update_buffer['actions'][ids[-1] + 1:],
                     dtype=action_dtype))
        rewards.append(
            np.array(update_buffer['rewards'][ids[-1] + 1:],
                     dtype=reward_dtype))
        done.append(
            np.array(update_buffer['done'][ids[-1] + 1:], dtype=done_dtype))

    return obs, actions, rewards, done


def _load_npz_demo(npz_path, use_visual_obs):
    '''
    Return:
        ndarray: visual_obs, actions, rewards, done

    E.g.
    (len, ...)
    '''
    data = np.load(npz_path)
    obs_name = 'visual_obs' if use_visual_obs else 'vector_obs'
    obs = data[obs_name]
    actions = data['actions']
    rewards = data['rewards']
    done = data['done']

    print("{}: {}, actions: {}, rewards: {}, done: {}".format(
        obs_name, obs.shape, actions.shape, rewards.shape, done.shape))

    return obs, actions, rewards, done
