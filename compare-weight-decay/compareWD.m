wd1 = dlmread('x_y1_y2_0.00025','\t');
wd1_x = wd1(:,1);
wd1_y1 = wd1(:,2);
wd1_y2 = wd1(:,3);

wd2 = dlmread('x_y1_y2_0.0005','\t');
wd2_x = wd2(:,1);
wd2_y1 = wd2(:,2);
wd2_y2 = wd2(:,3);

wd3 = dlmread('x_y1_y2_0.00075','\t');
wd3_x = wd3(:,1);
wd3_y1 = wd3(:,2);
wd3_y2 = wd3(:,3);

wd4 = dlmread('x_y1_y2_0.001','\t');
wd4_x = wd4(:,1);
wd4_y1 = wd4(:,2);
wd4_y2 = wd4(:,3);



figure,hold
plot(wd1_x(wd1_y1>0),wd1_y1(wd1_y1>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(wd2_x(wd2_y1>0),wd2_y1(wd2_y1>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(wd3_x(wd3_y1>0),wd3_y1(wd3_y1>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
plot(wd4_x(wd4_y1>0),wd4_y1(wd4_y1>0),'o-','Color','m','MarkerSize',5,...
    'MarkerEdgeColor','m');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('0.00025-train','0.00050-train','0.00075-train','0.00100-train')


figure,hold
plot(wd1_x(wd1_y2>0),wd1_y2(wd1_y2>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(wd2_x(wd2_y2>0),wd2_y2(wd2_y2>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(wd3_x(wd3_y2>0),wd3_y2(wd3_y2>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
plot(wd4_x(wd4_y2>0),wd4_y2(wd4_y2>0),'o-','Color','m','MarkerSize',5,...
    'MarkerEdgeColor','m');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('0.00025-test','0.00050-test','0.00075-test','0.00100-test')