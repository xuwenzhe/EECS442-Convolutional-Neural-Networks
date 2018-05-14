alpha1 = dlmread('x_y1_y2_0.7','\t');
alpha1_x = alpha1(:,1);
alpha1_y1 = alpha1(:,2);
alpha1_y2 = alpha1(:,3);

alpha2 = dlmread('x_y1_y2_0.8','\t');
alpha2_x = alpha2(:,1);
alpha2_y1 = alpha2(:,2);
alpha2_y2 = alpha2(:,3);

alpha3 = dlmread('x_y1_y2_0.9','\t');
alpha3_x = alpha3(:,1);
alpha3_y1 = alpha3(:,2);
alpha3_y2 = alpha3(:,3);




figure,hold
plot(alpha1_x(alpha1_y1>0),alpha1_y1(alpha1_y1>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(alpha2_x(alpha2_y1>0),alpha2_y1(alpha2_y1>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(alpha3_x(alpha3_y1>0),alpha3_y1(alpha3_y1>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('0.7-train','0.8-train','0.9-train')


figure,hold
plot(alpha1_x(alpha1_y2>0),alpha1_y2(alpha1_y2>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(alpha2_x(alpha2_y2>0),alpha2_y2(alpha2_y2>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(alpha3_x(alpha3_y2>0),alpha3_y2(alpha3_y2>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('0.7-test','0.8-test','0.9-test')