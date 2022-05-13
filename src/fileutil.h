#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>

namespace cave
{
    class FileException : public std::runtime_error
    {
        std::string message_;

    public:
        FileException(std::string message): std::runtime_error(message)
        {
           
        }
    };

    class Serializable
    {
    public:
        virtual void save(std::ostream &out) = 0;
        virtual void load(std::istream &in) = 0;
    };

    template <typename E>
    void saveValue(std::ostream &out, E &value)
    {
        out.write(reinterpret_cast<char *>(&value), sizeof(E));

        if (!out)
        {
            throw FileException("Error writing value to file.");
        }
    }

    template <typename E>
    E loadValue(std::istream &in)
    {
        E value;

        in.read(reinterpret_cast<char *>(&value), sizeof(E));

        if (!in)
        {
            throw FileException("Error reading value from file.");
        }

        return value;
    }

    template <typename E>
    void saveValueVector(std::ostream &out, std::vector<E> &vector)
    {
        int items = vector.size();
        cave::saveValue<int>(out, items);

        for (E &value : vector)
        {
            out.write(reinterpret_cast<char *>(&value), sizeof(E));
        }

        if (!out)
        {
            throw FileException("Error writing vector to file.");
        }
    }

    template <typename E>
    std::vector<E> loadValueVector(std::istream &in)
    {
        int items = loadValue<int>(in);

        std::vector<E> result;

        for (int i = 0; i < items; ++i)
        {
            E item = loadValue<E>(in);
            result.push_back(item);
        }

        if (!in)
        {
            throw FileException("Error reading vector from file.");
        }

        return result;
    }

    template <typename E>
    void saveSerializableVector(std::ostream &out, std::vector<E> &vector)
    {
        int items = vector.size();
        cave::saveValue<int>(out, items);

        for (E &value : vector)
        {
            value.save(out);
        }

        if (!out)
        {
            throw FileException("Error writing vector to file.");
        }
    }

    template <typename E>
    std::vector<E> loadSerializableVector(std::istream &in)
    {
        int items = loadValue<int>(in);

        std::vector<E> result;

        for (int i = 0; i < items; ++i)
        {
            E item;
            item.load(in);
            result.push_back(item);
        }

        if (!in)
        {
            throw FileException("Error reading vector from file.");
        }

        return result;
    }
}