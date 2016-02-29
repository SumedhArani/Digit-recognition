def get_data():
    """
    Get MNIST data ready to learn with.

    Returns
    -------
    dict
        With keys 'train' and 'test'. Both do have the keys 'X' (features)
        and'y' (labels)
    """
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')

    x = mnist.data
    y = mnist.target

    # Scale data to [-1, 1] - This is of major importance!!!
    x = x/255.0*2 - 1

    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
    data = {'train': {'X': x_train,
                      'y': y_train},
            'test': {'X': x_test,
                     'y': y_test}}
    return data

get_data()