#include <stdio.h>
#include <stdlib.h>
#include "svm.h"  // تضمين مكتبة libsvm

int main() {
    // إعداد بيانات الرسائل وعدد الكلمات وطول الرسالة (X) وتصنيفاتها (y)
    struct svm_problem problem;
    struct svm_parameter param;
    struct svm_model *model;
    
    // إعداد بيانات التدريب
    problem.l = 4;  // عدد البيانات
    problem.y = (double *)malloc(problem.l * sizeof(double));
    problem.x = (struct svm_node **)malloc(problem.l * sizeof(struct svm_node *));
    
    // البيانات (عدد كلمات معينة وطول الرسالة) وتصنيفاتها
    double labels[4] = {0, 1, 0, 1};  // 0 = عادية، 1 = غير مرغوب فيها
    double features[4][2] = {
        {4, 30},
        {10, 26},
        {2, 26},
        {6, 60}
    };
    
    // إعداد المتغيرات
    for (int i = 0; i < problem.l; i++) {
        problem.y[i] = labels[i];
        problem.x[i] = (struct svm_node *)malloc(3 * sizeof(struct svm_node));
        problem.x[i][0].index = 1;  // أول ميز ميزة = عدد الكلمات
        problem.x[i][0].value = features[i][0];
        problem.x[i][1].index = 2;  // ثاني ميز = طول الرسالة
        problem.x[i][1].value = features[i][1];
        problem.x[i][2].index = -1;  // نهاية الميزات
    }
    
    // إعداد المعاملات
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.C = 1;
    param.gamma = 0.5;

    // تدريب النموذج
    model = svm_train(&problem, &param);
    
    // الرسالة الجديدة التي نريد تصنيفها
    struct svm_node new_message[3];
    new_message[0].index = 1;
    new_message[0].value = 4;   // عدد كلمات معينة
    new_message[1].index = 2;
    new_message[1].value = 30;  // طول الرسالة
    new_message[2].index = -1;  // نهاية الميزات

    // تصنيف الرسالة الجديدة
    double classification = svm_predict(model, new_message);
    
    printf("تصنيف الرسالة الجديدة: %s\n", classification == 1 ? "غير مرغوب فيها" : "عادية");

    // تحرير الذاكرة
    svm_free_and_destroy_model(&model);
    free(problem.y);
    for (int i = 0; i < problem.l; i++) {
        free(problem.x[i]);
    }
    free(problem.x);

    return 0;
}
