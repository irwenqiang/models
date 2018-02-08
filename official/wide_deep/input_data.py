import tensorflow as tf

def create_feature_columns():
    # Continuous columns.
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    # Sparse columns.
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        "education",[
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
        ])
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        "marital_status", [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'
        ])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        "relationship", [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'
        ])
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        "workclass", [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'
        ])
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000)

    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[
        18, 25, 30, 35, 40, 45, 50, 55, 60, 65
    ])

    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        "gender", [
            "female", "male"
        ])
    race = tf.feature_column.categorical_column_with_vocabulary_list(
        "race", [
            "Amer-Indian-Eskimo",
            "Asian-Pac-Islander",
            "Black",
            "Other",
            "White"
        ])

    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000
        ),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
        )
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass)
    ]


    label = tf.feature_column.numeric_column("label", dtype=tf.int64)

    return wide_columns, deep_columns


def input_fn(mode, data_file, batch_size):
    wide_features, deep_features = create_feature_columns()
    input_features = wide_features + deep_features
    features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

    feature_map = tf.contrib.learn.io.read_batch_record_features(
        file_pattern=[data_file],
        batch_size=batch_size,
        features=features,
        name="read_batch_features_{}".format(mode))

    target = feature_map.pop("label")

    return feature_map, target
