def main():
    # 0. get started
    print("\nBegin GAN MNIST demo ")
    np.random.seed(1)

    # 1. create MNIST DataLoader object
    print("\nCreating MNIST Dataset and DataLoader ")

    bat_size = 64
    # requested size, not necessarily actual
    # 60,000 train images / 64 = 937 batches + 1 of size 32

    trfrm = tv.transforms.ToTensor()  # also divides by 255
    train_ds = tv.datasets.MNIST(root="data", train=True,
                                 download=True, transform=trfrm)
    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=bat_size, shuffle=True, drop_last=True)

    # 2. create networks
    dis = Discriminator().to(device)  # 784-128-64-32-1
    gen = Generator().to(device)  # 100-32-64-128-784

    # 3. train GAN model
    max_epochs = 100
    ep_log_interval = 10
    lrn_rate = 0.002  # small for Adam

    dis.train()  # set mode
    gen.train()
    dis_optimizer = T.optim.Adam(dis.parameters(), lrn_rate)
    gen_optimizer = T.optim.Adam(gen.parameters(), lrn_rate)
    loss_func = T.nn.BCELoss()
    all_ones = T.ones(bat_size, dtype=T.float32).to(device)
    all_zeros = T.zeros(bat_size, dtype=T.float32).to(device)

    # -----------------------------------------------------------

    print("\nStarting training ")
    for epoch in range(0, max_epochs):
        for batch_idx, (real_images, _) in enumerate(train_ldr):
            dis_accum_loss = 0.0  # to display progress
            gen_accum_loss = 0.0

            # 1a. train discriminator (0/1) using real images
            dis_optimizer.zero_grad()
            dis_real_oupt = dis(real_images)  # [0, 1]
            dis_real_loss = loss_func(dis_real_oupt.squeeze(),
                                      all_ones)

            # 1b. train discriminator using fake images
            zz = T.normal(0.0, 1.0,
                          size=(bat_size, 100)).to(device)
            fake_images = gen(zz)
            dis_fake_oupt = dis(fake_images)
            dis_fake_loss = loss_func(dis_fake_oupt.squeeze(),
                                      all_zeros)
            dis_loss_tot = dis_real_loss + dis_fake_loss
            dis_accum_loss += dis_loss_tot

            dis_loss_tot.backward()
            dis_optimizer.step()

            # 2. train gen with fake images and flipped labels
            gen_optimizer.zero_grad()
            zz = T.normal(0.0, 1.0,
                          size=(bat_size, 100)).to(device)
            fake_images = gen(zz)
            dis_fake_oupt = dis(fake_images)
            gen_loss = loss_func(dis_fake_oupt.squeeze(), all_ones)
            gen_accum_loss += gen_loss

            gen_loss.backward()
            gen_optimizer.step()

        if epoch % ep_log_interval == 0 or epoch == max_epochs - 1:
            print(" epoch: %4d | dis loss: %0.4f | gen loss: %0.4f " \
                  % (epoch, dis_accum_loss, gen_accum_loss))
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            fn = ".\\Models\\" + str(dt) + str("-") + "epoch_" + \
                 str(epoch) + "_gan_mnist_model.pt"
            T.save(gen.state_dict(), fn)

    print("Training commplete ")

    # -----------------------------------------------------------

    print("\nGenerating 16 images using trained generator ")
    num_images = 16
    zz = T.normal(0.0, 1.0, size=(num_images, 100)).to(device)
    gen.eval()
    rand_images = gen(zz)
    images_list = [rand_images]
    view_images(images_list, 0)

    print("\nEnd GAN MNIST demo ")


if __name__ == "__main__":
    main()