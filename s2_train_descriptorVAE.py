

D = 17
M = 256
L = 8

encoder = Encoder(EncoderMLP(D, L, M))
decoder = Decoder(DecoderMLP(D, L, M))

prior = MoGPrior(L=L, num_components=16)

model = VAE(encoder=encoder, decoder=decoder, prior=prior, L=L)