---
layout: post
title: Abusing hash kernels for wildly unprincipled machine learning
description: >
  Hash kernels were originally designed for learning on
  bag-of-words document representations. But with a little
  imagination, we can use them to learn on all types of
  structured data.

tags: ["ml"]
---
{% include JB/setup %}

*Want to skip to the code? Find it on
[Github](https://github.com/jeremydhoon/hashkernel).*

Modern [machine learning](http://en.wikipedia.org/wiki/Support_vector_machine)
[techniques](http://en.wikipedia.org/wiki/
Deep_learning#Deep_Learning_in_Artificial_Neural_Networks)
are so powerful that the hardest part of training a
classifier is often getting training data into the right format. What if
we could skip all the tedious preparation and train a classifier on
virtually any kind of data? In this post, I'll go
over the trouble with preparing data for machine learning, and then describe
[hashkernel](https://github.com/jeremydhoon/hashkernel/),
a Python module I wrote to demonstrate how a technique called *hash kernels*
can avoid these pitfalls.

While I hope that this post will be accessible to readers without any
experience with machine learning, you may find it easier to follow along if
you're already comfortable with concepts such as
[classification](http://en.wikipedia.org/wiki/Statistical_classification) and
[logistic regression](http://en.wikipedia.org/wiki/Logistic_regression).


No Classification Without Representation
----------------------------------------

*Note: if you're familiar with basic machine learning concepts and with
the bag-of-words model, please feel free to skip to the next section.*

Learning algorithms all require their training data appear in a specific
format. While the format may vary from one algorithm to another, most
algorithms take as input a
*[feature vector](http://en.wikipedia.org/wiki/Feature_vector)*,
an array containing values all
of the same type. Feature vectors are a great way to represent some inputs,
like a geometric point in space where each entry in the feature vector is a
coordinate, or a fixed-size image where each entry is the brightness of a
pixel.

Yet data in real-world applications is much messier. For starters, the
features in a real-world dataset rarely all share the same data type.
Fields present in one
[instance](http://robotics.stanford.edu/~ronnyk/glossary.html)
may be missing in another. To easily training a classifier on real world data,
we need to be able to convert data in an arbitrary format consisting of
mixed of types into a feature vector that a classifier can understand.

Computer scientists working in the area of *text classification*
understand this problem quite well, since instances in this area are simply
strings of text -- a format far different from traditional feature vectors.
For example, one might wish to classify an email message
as "spam" or "ham". In order to do so, the text of each email message must be
converted to a feature vector. This is typically achieved by applying a
transformation know as the
*[bag-of-words model](http://en.wikipedia.org/wiki/Bag-of-words_model)*.

In bag-of-words, we assign an index to each distinct word that appears in
any training instance.
The resulting feature vector has length
*max_word_index + 1*. Each entry in the feature vector is either one
of the word for the entry's index is present, or zero otherwise.

Bag-of-words can be unwieldy.
First, we might need to know every English word.
What about commonly-used proper nouns that don't appear in the
[Oxford English Dictionary](http://en.wikipedia.org/wiki/Oxford_English_Dictionary)?
Next, we take
note of the sparsity of the bag-of-words feature vector. For example, in the
problem of separating spam- and ham emails, each message would typically
contain no more than a few hundred words. Yet there are over one hundred
thousand unique words in the OED alone.

Introducing Hash Kernels
------------------------

Hash kernels were invented to surmount both of these problems with bag-of-words.

In this section, I'll first describe how hash kernels work, and then how
they can help us more efficiently convert a piece of input text to a
feature vector suitable for machine learning.

Fundamentally, a hash kernel is just a fixed-size array.
Unlike in the traditional bag-of-words model from the previous
section, the size of the array isn't rigidly determined by the input space.
In a text classification problem like spam filtering, this means that the size
of the array isn't determined by the number of distinct words under
consideration.
Instead, hash kernels give us the freedom to choose an array size that
balances computational performance with classification accuracy (more on
how this works in a bit).

The "hash" part of the name "hash kernel" comes from the way features are
incorporated into the array. Using a
[hash function](http://en.wikipedia.org/wiki/Hash_function), we map each
feature to an index in the array. In the spam filtering case, where each word
is a feature (one if the word is found in an input text, or zero if absent),
we hash that word to obtain an index in the array. We then add the
value of the feature to the array entry at that index, with a few twists.
[Weinberg et al.](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf)
suggest adding or subtracting the value of the feature according to
another hash function. Here's a more concrete description of
Weinberg's method, expressed in Python:

    feature_vector = [0 for _ in xrange(1 << bits)]
    for word in input_text.split():
      hash = crc32(word)  # CRC-32 hash function, available in the zlib module
      index = hash & ((1 << bits) - 1)
      # Use the nth bit, zero-indexed, to determine if we add or subtract
      # one from the index.
      sign = (((hash & (1 << bits)) >> bits) << 1) - 1
      feature_vector[index] += sign

The final step is to feed the array into a learner to create a classifier.
Since the array looks just like a typical feature vector, it should be
compatible with all your favorite learning algorithms.

Brace For Impact
----------------

If you're following along closely, you may have noticed that some words
may "collide" on the same index. That is, a single
hash kernel array element may contain information about more than one feature.
Empirically, a
moderate collision rate tends to have a benign effect on classification
performance.
[Shi et al.](http://jmlr.csail.mit.edu/papers/volume10/shi09a/shi09a.pdf)
observe that in one of their experiments, a collision rate of 94% (that is,
94% of features mapped to the same index as one or more other features) resulted
in an increase in the experimental error rate from 5.5% to only 6%.

But as the frequency of collisions reaches extreme levels, where each array
element plays host to many features, classification accuracy inevitably suffers.
When this happens, we can increase the size of the hash kernel array, so that
these features are spread out of over more elements.

If We Can Hash It, We Can Learn It
----------------------------------

Now that we've seen how hash kernels can power text
classifiation tasks, let's take a look at how we can apply hash kernels to
just about any type of structured data. All we need is a hash function for
that takes as input a piece of structured data and maps its pieces to
indices in a hash kernel array. This opens possibilities
ranging from learning on a stream of arbitrary JSON data, to classifying
objects in a dynamic programming language like Python or Javascript.

To demonstrate how hash kernels can learn on arbitrary structured data, I put
together [hashkernel](https://github.com/jeremydhoon/hashkernel),
a hash-kernel-based online
learner. Feed it a stream of arbitrary Python data structures (any mixture
of lists, dictionaries, or even objects) and it will train on these inputs.
It does so by recursively exploring each
instance and treating each path in this recursion as if it were a word in
a bag-of-words, as with email messages in a spam detection task.

Hashkernel Classification Performance
-------------------------------------

As an initial test of hashkernel, I obtained two datasets from
the [UC Irvine Machine Learning Dataset Repository](
http://archive.ics.uci.edu/ml/datasets/). With virtually no set-up, I was able
to train a classifier on each dataset and achieve respectable
results, in line with those reported by researchers using other, more
labor-intensive machine learning techniques. You can replicate these results
by running *hashkernel.py*:

    $ python ./hashkernel.py --mushroom --us-adult -b 16 -t0.25
    Error rate on load_mushroom_dataset: 0.178
    Error rate on load_us_adult_dataset: 0.171

For comparison, the contributors of the
[US Adult dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
error rates no better than fourteen percent using a number of different machine
learning techniques.

These datasets were designed to be easy to work with. They're neatly
arranged in [CSV](http://en.wikipedia.org/wiki/Comma-separated_values) format
to be especially easy to feed into a learning algorithms. But I designed
hashkernel to work with complex, messy structured data.

To show off hashkernel's versatility, I obtained data for 346 pieces of
data taken from my personal Facebook account, using
Facebook's [Graph API](https://developers.facebook.com/). My goal was to train
a classifier that predicts whether a piece of Facebook content will receive
at least one Like. The structured data I gathered includes
information about my status updates, photos I'm tagged in, and
stories in my newsfeed. Each piece of data consists of a
JSON dictionary. Each dictionary contains nested information,
such as comments, likes, and
[CDN](http://en.wikipedia.org/wiki/Content_delivery_network) URLs for
associated images. The following example has been edited to remove
irrelevant attributes such as URLs for alternate image sizes, and to preserve
my friends' privacy.

<pre style="height:16em; overflow:scroll;">
{
    "picture": "[URL]",
    "from": {
	"name": "[friend's name]",
	"id": "[friend's FB UID]"
    },
    "name": "The North Shore of Lake Tahoe!",
    "source": "[URL]",
    "comments": {
	"data": [
	    {
		"from": {
		    "name": "[different friend's name]",
		    "id": "[different friend's FB UID]"
		},
		"like_count": 1,
		"can_remove": false,
		"created_time": "2012-03-30T01:29:00+0000",
		"message": "We are so cute!",
		"id": "[friend's FB UID]",
		"user_likes": false
	    }
	]
    },
    "height": 540,
    "width": 720,
    "likes": {
	"data": [
	    {
		"id": "31062",
		"name": "Jeremy Hoon"
	    }
	]
    },
    "created_time": "2012-03-30T00:29:15+0000",
    "updated_time": "2012-03-30T00:29:17+0000",
    "id": "3414272829923",
}
</pre>

To obtain training labels, I examined and stripped the "likes" attribute from
each object's dictionary (where a lack of "likes"
attribute was interpreted as zero likes). Using these stripped dictionaries
as instances, I was able
to train a hashkernel classifier with little effort:

    import hashkernel

    def is_liked(item):
        if "likes" not in item:
            return False
        elif "count" not in item["likes"]:
            item["likes"]["count"] = len(item["likes"]["data"])
        likes = item.pop("likes")
        return likes["count"] > 0

    def train(train_set):
        kernel = hashkernel.HashKernelLogisticRegression(bits=17)
        for instance in train_set:
	    label = is_liked(instance)
            kernel.add(instance, label)
	return kernel

To evaluate this classifier, I ran leave-one-out cross-validation over the
set of 346 Facebook data objects. I ran the experiment using seventeen bits of
hash space, meaning that I used an array size of 2^17 elements.
You can find my implementation in
[fblearn.py](https://github.com/jeremydhoon/hashkernel/blob/master/fblearn.py).
To replicate my results, run

    $ python fblearn.py -b17
    failure rate: 0.260115606936
    fp/fn/total 40 50 346
    rate liked: 0.439306358382

This output indicates an error rate of twenty six percent, meaning that
hashkernel was correct about 74% of the time. For comparison, always guessing
"not liked" would have resulted in an accuracy of 56%.

I'm sure I could have done a better job of classifying my Facebook data
with the help of some intelligent pre-processing (for example, by
bucketing numeric features). But considering the ease of building a hashkernel
classifier, I am quite happy with this figure.

Further Reading
---------------

The academic literature contains a wealth of information on the (principled)
use of hash kernels:

- [Weinberg et al.](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf)
- [Shi et al.](http://hunch.net/~jl/projects/hash_reps/hash_kernels/hashkernel.pdf)

Some of this academic work has been implemented in the featureful
[Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki)
package out of Yahoo- and Microsoft's research arms. It
further exploits hash kernels by using the hash to map words (or other
types of features) to different machines in a cluster in order to parallelize
the machine learning process.