
import torch
import torchvision
import torchvision.transforms as transforms
import time

import net_functions as nf
import cnfge

###################################################################################################

prep_start_time = time.time()

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 50

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False)

###################################################################################################

ADN = {
        "x1":32,"x2":64,"x3":128,
        "y1":3,"y2":3,"y3":3,
        "z1":64,"z2":64,
        "BATCH_SIZE":batch_size,"lr":0.001 
        } 

net = nf.create_net(ADN)
optimizer = net.optimizer
criterion = net.loss_function

prep_end_time = time.time()

###################################################################################################

train_start_time = time.time()

for epoch in range(cnfge.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(cnfge.device)
        labels = labels.to(cnfge.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        frequency = 100
        running_loss += loss.item()
        if i % frequency == (frequency-1):    # print and write every 100 mini-batches

            print('[%d, %5d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / frequency))

            f = open('mnist_loss_' + net.model_name + '.txt', "a")
            f.write(f"{net.model_name}, {time.time()}, {epoch+1}, {i+1}, {running_loss / frequency} \n")
            f.close()

            running_loss = 0.0


    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(cnfge.device)
            labels = labels.to(cnfge.device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('[%d, %5d] Accuracy: %.3f %%' % (epoch+1, i+1, 100*correct/total))

    f = open('mnist_acc_' + net.model_name + '.txt', "a")
    f.write(f"{net.model_name}, {time.time()}, {epoch+1}, {i+1}, {100*correct/total} \n")
    f.close()
    
    train_end_time = time.time()

print('Finished Training')

f = open('mnist_times.txt', "a")
f.write(f"{net.model_name}, {prep_start_time}, {prep_end_time}, {train_start_time}, {train_end_time} \n")
f.close()


PATH = './MNIST_' + net.model_name + '.pth'
torch.save(net.state_dict(), PATH)