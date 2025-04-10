from numpy import uint16, ceil, linspace
from pandas import DataFrame, MultiIndex
from itertools import product
import synthwave
import matplotlib.pylab as plt
import warnings


class Ageing:
    def __init__(
        self,
        population_pyramid: DataFrame,
        mortality_rates: DataFrame,
        birth_numbers: DataFrame,
        start_year: uint16 = uint16(2011),
        end_year: uint16 = uint16(2020),
    ):

        self.pop = population_pyramid
        self._validate_population_pyramid()

        self.mortality = mortality_rates
        self._validate_mortality_rates()

        self.birth_numbers = birth_numbers
        self._validate_birth_numbers()

        self.start_year = start_year
        self.end_year = end_year
        self._validate_dates()

        indices = range(1, 2)

        self.index = [
            (year_, sex, age, country_name) for year_, sex, age, country_name in indices
        ]

        mi = MultiIndex.from_tuples(
            self.index, names=("year", "sex", "age_group", "country")
        )
        columns = ["people_total", "mortality_rate", "dead_within_year"]
        self.df = DataFrame(index=mi, columns=columns)

    def _validate_dates(self) -> None:
        if not isinstance(self.start_year, uint16):
            raise TypeError("Starting point is not a uint16")

        if not isinstance(self.end_year, uint16):
            raise TypeError("Ending point is not a uint16")

        if self.start_year != 2011:
            warnings.warn(
                f"Warning: current code considers only year 2011 as a"
                f" valid starting point"
            )

        if self.end_year <= self.start_year:
            raise ValueError("Invalid order of years")

    def _validate_population_pyramid(self) -> None:
        if not isinstance(self.pop, DataFrame):
            raise TypeError(
                "Population pyramid must be supplied in the form of a DataFrame"
            )

    def _validate_mortality_rates(self) -> None:
        if not isinstance(self.mortality, DataFrame):
            raise TypeError(
                "Mortality must be supplied in the form of a DataFrame"
            )

    def _validate_birth_numbers(self) -> None:
        if not isinstance(self.birth_numbers, DataFrame):
            raise TypeError(
                "Birth numbers must be supplied in the form of a DataFrame"
            )

    def _increment_age(self, df: DataFrame) -> DataFrame:
        pass

    def _add_new_people(self, df: DataFrame) -> DataFrame:
        pass

    def _remove_dead_people(self):
        pass

    def run(self) -> DataFrame:
        # populate year 2011
        self.df["people_total"] = self.pop["people_total"]

        # add all mortality rates for all years
        self.df["mortality_rate"] = self.mortality["mortality_rate"]

        for year_ in range(2011, (self.end_year + 1)):
            # calculate the number of dead people
            # do we round up or down?
            dead = self.df.loc[year_, ["people_total"]]
            dead *= self.df.loc[year_, ["mortality_rate"]].values

            dead /= 100_000
            dead = dead.apply(lambda row: ceil(row)).astype("uint32")
            self.df.loc[year_, "dead_within_year"] = dead.values

            # calculate pop - deaths next year and age it
            # at the moment all people who reach 100 die immediately.
            pop_next_year = self.df.loc[year_, ["people_total"]]
            pop_next_year -= self.df.loc[year_, ["dead_within_year"]].values
            pop_next_year = pop_next_year.groupby(level=[0, 2]).shift(1)

            births_next_year = self.birth_numbers.loc[year_]
            births_next_year.rename(
                columns={"birth_number": "people_total"}, inplace=True
            )

            pop_next_year.update(births_next_year)

            self.df.loc[[year_ + 1], "people_total"] = pop_next_year[
                "people_total"
            ].values

        return self.df


if __name__ == "__main__":
    cen = synthwave.load_census_data()
    mo = synthwave.load_mortality_rates()
    b = synthwave.load_birth_numbers()
    a = Ageing(cen, mo, b)
    data = a.run()
    data.to_csv("result.csv")

    plt.figure()
    colors = plt.cm.jet(linspace(0, 1, 44))

    for i, year in enumerate(range(2011, (2020 + 1))):
        data.loc[(year, "f", slice(None), "e")]["people_total"].plot(
            color=colors[i],
            figsize=(16, 12),
            fontsize=30,
            label="e " + str(year),
            logy=True,
        )

    for i, year in enumerate(range(2011, (2020 + 1))):
        data.loc[(year, "f", slice(None), "w")]["people_total"].plot(
            color=colors[i + 11], figsize=(16, 12), fontsize=30, label="w " + str(year)
        )

    for i, year in enumerate(range(2011, (2020 + 1))):
        data.loc[(year, "f", slice(None), "s")]["people_total"].plot(
            color=colors[i + 22], figsize=(16, 12), fontsize=30, label="s " + str(year)
        )

    for i, year in enumerate(range(2011, (2020 + 1))):
        data.loc[(year, "f", slice(None), "ni")]["people_total"].plot(
            color=colors[i + 33], figsize=(16, 12), fontsize=30, label="ni " + str(year)
        )

    plt.grid(axis="both", color="0.95")
    plt.legend(title="Year of study:")
    plt.title("Number of people per age group")
    # plt.show()
    plt.savefig("result.png")
