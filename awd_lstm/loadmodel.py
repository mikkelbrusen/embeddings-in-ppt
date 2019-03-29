import torch

def replace_params(module_old, module_new):
	for name_old, param_old in module_old.named_parameters():
		for name_new, param_new in module_new.named_parameters():
			if name_old == name_new:
				print(name_old)
				param_old.data = param_new.data



def load_params(model, params="awd_lstm/test_v2.pt"):
	params_model = None
	with open(params, 'rb') as f:
		params_model, _, _ = torch.load(f, map_location='cpu')

	# old
	old_rnns = model.rnns
	old_enc = model.encoder
	old_dec = model.decoder

	# new 
	new_rnns = params_model.rnns
	new_enc = params_model.encoder
	new_dec = params_model.decoder

	# Overwrite embedding
	print("Overwriting embedding")
	replace_params(old_enc, new_enc)
	
	# Overwrite rnns
	print("Overwriting rnns")
	replace_params(model.rnns[0], params_model.rnns[0].module)
	replace_params(model.rnns[1], params_model.rnns[1].module)
	replace_params(model.rnns[2], params_model.rnns[2].module)

	# Overwrite decoder
	print("Overwriting decoder")
	replace_params(old_dec, new_dec)