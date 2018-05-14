eta1 = dlmread('x_y1_y2_0.02','\t');
eta1_x = eta1(:,1);
eta1_y1 = eta1(:,2);
eta1_y2 = eta1(:,3);

eta2 = dlmread('x_y1_y2_0.05','\t');
eta2_x = eta2(:,1);
eta2_y1 = eta2(:,2);
eta2_y2 = eta2(:,3);

eta3 = dlmread('x_y1_y2_0.1','\t');
eta3_x = eta3(:,1);
eta3_y1 = eta3(:,2);
eta3_y2 = eta3(:,3);

eta4 = dlmread('x_y1_y2_0.15','\t');
eta4_x = eta4(:,1);
eta4_y1 = eta4(:,2);
eta4_y2 = eta4(:,3);

eta5 = dlmread('x_y1_y2_0.2','\t');
eta5_x = eta5(:,1);
eta5_y1 = eta5(:,2);
eta5_y2 = eta5(:,3);

figure,hold
plot(eta1_x(eta1_y1>0),eta1_y1(eta1_y1>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(eta2_x(eta2_y1>0),eta2_y1(eta2_y1>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(eta3_x(eta3_y1>0),eta3_y1(eta3_y1>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
plot(eta4_x(eta4_y1>0),eta4_y1(eta4_y1>0),'o-','Color','m','MarkerSize',5,...
    'MarkerEdgeColor','m');
plot(eta5_x(eta5_y1>0),eta5_y1(eta5_y1>0),'o-','Color','c','MarkerSize',5,...
    'MarkerEdgeColor','c');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('0.02-train','0.05-train','0.10-train','0.15-train','0.20-train')


figure,hold
plot(eta1_x(eta1_y2>0),eta1_y2(eta1_y2>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(eta2_x(eta2_y2>0),eta2_y2(eta2_y2>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(eta3_x(eta3_y2>0),eta3_y2(eta3_y2>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
plot(eta4_x(eta4_y2>0),eta4_y2(eta4_y2>0),'o-','Color','m','MarkerSize',5,...
    'MarkerEdgeColor','m');
plot(eta5_x(eta5_y2>0),eta5_y2(eta5_y2>0),'o-','Color','c','MarkerSize',5,...
    'MarkerEdgeColor','c');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('0.02-test','0.05-test','0.10-test','0.15-test','0.20-test')