{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Required imports"
      ],
      "metadata": {
        "id": "EiAvLe1V0FaH"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "V_3j101Zz9Mk"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Load the Data"
      ],
      "metadata": {
        "id": "H64A0fi80kmA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the training data\n",
        "train_data = pd.read_excel('train.xlsx')\n",
        "\n",
        "# Load the testing data\n",
        "test_data = pd.read_excel('test.xlsx')\n"
      ],
      "metadata": {
        "id": "Rr-eLRdb0lFX"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Prepare the data"
      ],
      "metadata": {
        "id": "jGraLezL0pNw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = train_data.iloc[:, :-1]  # All columns except the last one\n",
        "y_train = train_data.iloc[:, -1]   # The last column\n",
        "\n",
        "X_test = test_data.iloc[:, :-1]\n",
        "y_test = test_data.iloc[:, -1]"
      ],
      "metadata": {
        "id": "LAgCLjn90ssn"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Classification model\n"
      ],
      "metadata": {
        "id": "4j4Y89wh0x6_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',max_depth = 27, random_state = 42,max_features='sqrt',bootstrap= True)\n",
        "classifier.fit(X_train, y_train)\n",
        "y_pred = classifier.predict(X_test)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Oj7AYUhb02LX",
        "outputId": "5137c8f4-b512-4a29-d403-27a9aa4c6a15"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluation of the model\n"
      ],
      "metadata": {
        "id": "4o0tVcjw1ty3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Making the Confusion Matrix\n",
        "from sklearn.metrics import confusion_matrix\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "\n",
        "accuracy = accuracy_score(y_test,y_pred)\n",
        "\n",
        "# Affichez la précision du modèle\n",
        "print(\"Accuracy of Randomforest :\", accuracy)\n",
        "report = classification_report(y_test, y_pred)\n",
        "print(\"Report of classification model :\\n\", report)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7EZsSlzz08kH",
        "outputId": "e20be574-556b-47e2-bac6-525cb2ab96f4"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of Randomforest : 0.8860759493670886\n",
            "Report of classification model :\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "           0       0.83      0.71      0.77         7\n",
            "           1       0.88      0.88      0.88         8\n",
            "           2       1.00      0.75      0.86         8\n",
            "           3       1.00      1.00      1.00         8\n",
            "           4       0.89      1.00      0.94         8\n",
            "           5       0.89      1.00      0.94         8\n",
            "           6       0.88      0.88      0.88         8\n",
            "           7       0.86      0.75      0.80         8\n",
            "           8       1.00      0.88      0.93         8\n",
            "           9       0.73      1.00      0.84         8\n",
            "\n",
            "    accuracy                           0.89        79\n",
            "   macro avg       0.89      0.88      0.88        79\n",
            "weighted avg       0.90      0.89      0.88        79\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Confusion Matrics"
      ],
      "metadata": {
        "id": "IVKwwWdR1nCX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm)\n",
        "disp.plot()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 455
        },
        "id": "jVSkkooF1AV7",
        "outputId": "7a16f630-db68-4880-d483-80f202556940"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}
