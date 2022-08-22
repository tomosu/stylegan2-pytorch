import torch
from torchvision import transforms, utils

# https://arxiv.org/abs/2010.05334


def extract_conv_names(model, resolution):

  extract_names = {
      4: [],
      8: ['convs.0.', 'convs.1.', 'to_rgbs.0.'],
      16: ['convs.0.', 'convs.1.', 'to_rgbs.0.', 'convs.2.', 'convs.3.', 'to_rgbs.1.'],
      32: ['convs.0.', 'convs.1.', 'to_rgbs.0.', 'convs.2.', 'convs.3.', 'to_rgbs.1.', 'convs.4.', 'convs.5.', 'to_rgbs.2.'],
      64: ['convs.0.', 'convs.1.', 'to_rgbs.0.', 'convs.2.', 'convs.3.', 'to_rgbs.1.', 'convs.4.', 'convs.5.', 'to_rgbs.2.',
           'convs.6.', 'convs.7.', 'to_rgbs.3.']
  }

  # input    conv1    to_rgb1    convs   to_rgbs
  keys = [key for key, value in model.items()]
  used_names = []
  for key in keys:
    if 'input' in key:
      used_names.append((key, 0))
    elif 'conv1' in key:
      used_names.append((key, 0))
    elif 'to_rgb1' in key:
      used_names.append((key, 0))
    elif 'convs' in key:
      temp_label = True
      for cn in extract_names[resolution]:
        if cn in key:
          used_names.append((key, 0))
          temp_label = False
      if temp_label:
        used_names.append((key, 1))
    elif 'to_rgbs' in key:
      temp_label = True
      for cn in extract_names[resolution]:
        if cn in key:
          used_names.append((key, 0))
          temp_label = False
      if temp_label:
        used_names.append((key, 1))

  return used_names


def blend_models(model_1, model_2, resolution):
  # y is the blending amount which y = 0 means all model 1, y = 1 means all model_2

  model_1_names = extract_conv_names(model_1, resolution)
  model_2_names = extract_conv_names(model_2, resolution)

  assert all((x == y for x, y in zip(model_1_names, model_2_names)))

  model_out = model_1.copy()

  model_names = [x[0] for x in model_1_names]
  model_y = [x[1] for x in model_1_names]

  for key, y in zip(model_names, model_y):
    model_out[key] = model_1[key] * (1 - y) + model_2[key] * y

  return model_out
