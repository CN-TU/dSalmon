// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_PLACEHOLDERQUEUE_H
#define DSALMON_PLACEHOLDERQUEUE_H

#include <functional>
#include <vector>
#include <boost/heap/d_ary_heap.hpp>

template<typename Key, typename T, typename Tag>
class PlaceholderQueue {
    struct Placeholder {
        Key key;
        Tag tag;
        std::size_t multiplicity;
    };
    struct Entry {
        Key key;
        T value;
    };
    
    typedef std::function<bool(const Key&, const Key&)> CompareFunction;
    CompareFunction compare;
    Key max_key;
    
    // TODO: make this cleaner (without std::bind())
    static bool placeholder_cmp(CompareFunction& cmp, const Placeholder& a, const Placeholder& b) {
        return cmp(a.key, b.key);
    }
    static bool entry_cmp(CompareFunction& cmp, const Entry& a, const Entry& b) {
        return cmp(a.key, b.key);
    }
    
    typedef boost::heap::d_ary_heap<
        Placeholder,
        boost::heap::compare<std::function<bool(const Placeholder&, const Placeholder&)>>,
        boost::heap::mutable_<true>,
        boost::heap::arity<2>
    > PlaceholderHeap;
    typedef typename PlaceholderHeap::handle_type PlaceholderHandle;
    
    PlaceholderHeap placeholder_heap;
    
    std::vector<PlaceholderHandle> tag_table;
    std::vector<Tag> tag_table_tags;
    
    std::size_t max_size;
    std::size_t placeholder_total;
    
    std::size_t tagTableHash(Tag tag) {
        return std::hash<Tag>{}(tag) % tag_table.size();
    }
    
    void removeFromTagTable(Tag tag) {
        if (tag != Tag{}) {
            std::size_t hash = tagTableHash(tag);
            if (tag_table_tags[hash] == tag)
                tag_table_tags[hash] = Tag{};
        }
    }
    
    void reducePlaceholder(PlaceholderHandle handle, std::size_t multiplicity) {
        Placeholder& placeholder = *handle;
        if (placeholder.multiplicity <= multiplicity) {
            placeholder_total -= placeholder.multiplicity;
            removeFromTagTable(placeholder.tag);
            placeholder_heap.erase(handle);
        }
        else {
            placeholder_total -= multiplicity;
            placeholder.multiplicity -= multiplicity;
        }
    }
    
    void makeRoom(Key upper_bound, std::size_t to_free) {
        std::size_t remaining = to_free;
        while (!placeholder_heap.empty() && compare(upper_bound, placeholder_heap.top().key)) {
            if (placeholder_heap.top().multiplicity <= remaining) {
                remaining -= placeholder_heap.top().multiplicity;
                removeFromTagTable(placeholder_heap.top().tag);
                placeholder_heap.pop();
                if (remaining == 0)
                    break;
            }
            else {
                // TODO: this. is. ugly!
                Placeholder& placeholder = const_cast<Placeholder&>(placeholder_heap.top());
                placeholder.multiplicity -= remaining;
                remaining = 0;
                break;
            }
        }
        placeholder_total -= to_free - remaining;
    }
    
public:
    PlaceholderQueue() {};
    // TODO: pass compare by reference or std::move ?
    PlaceholderQueue(std::size_t max_size, CompareFunction compare, Key max_key) :
        compare(compare), max_key(max_key), 
        placeholder_heap(std::bind(placeholder_cmp, this->compare, std::placeholders::_1, std::placeholders::_2)),
        tag_table(2*max_size), tag_table_tags(2*max_size),
        max_size(max_size), placeholder_total(0)
    {
        placeholder_heap.reserve(max_size);
    }
    
    PlaceholderQueue& operator=(const PlaceholderQueue&) = default;
    
    void addPlaceholder(Tag from_placeholder, Key upper_bound, std::size_t multiplicity, Tag tag) {

        if (from_placeholder != Tag{}) {
            std::size_t from_placeholder_hash = tagTableHash(from_placeholder);
            if (tag_table_tags[from_placeholder_hash] == from_placeholder) {
                reducePlaceholder(tag_table[from_placeholder_hash], multiplicity);
            }
        }
        makeRoom(upper_bound, std::max<std::size_t>(0, multiplicity - (max_size - placeholder_total)));
        std::size_t new_multiplicity = std::min(multiplicity, max_size-placeholder_total);
        auto new_handle =
            placeholder_heap.push(Placeholder{upper_bound, tag, new_multiplicity});
        placeholder_total += new_multiplicity;
        if (tag != Tag{}) {
            std::size_t tag_hash = tagTableHash(tag);
            if (tag_table_tags[tag_hash] != Tag{}) {
                placeholder_total -= (*tag_table[tag_hash]).multiplicity;
                placeholder_heap.erase(tag_table[tag_hash]);
            }
            tag_table_tags[tag_hash] = tag;
            tag_table[tag_hash] = new_handle;
        }
    }
    
    Key getMaxKey() { 
        if (placeholder_heap.size() < max_size)
            return max_key;
        else
            return placeholder_heap.top().key;
    }
};

#endif
