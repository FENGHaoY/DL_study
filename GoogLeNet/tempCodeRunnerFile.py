device = torch.device('cuda')
    # with torch.no_grad():
    #     model.to(device)
    #     for index,(x,y) in enumerate(test_dl):
    #         x = x.to(device)
    #         y = y.to(device)
    #         model.eval()
    #         output = model(x)
    #         res = torch.argmax(output,dim=1)
    #         for i in range(len(res)):
    #             print(f"预测值：{res[i].item()}, 实际值{y[i].item()}")